# app/services/memory_service.py
"""
Memory, stored in MySQL under django_euf_admin — never on the agent's disk.

Hermes' native model keeps MEMORY.md and USER.md as files inside the profile
home. That is unacceptable here: a user profile is personal data on an EU
platform, it has to be in the database with the rest of the account, subject to
the same access control, export and deletion as everything else. So the built-in
memory is disabled in config.yaml (`memory_enabled: false`,
`user_profile_enabled: false`) and this module stands in for it, over the chat
endpoints django_euf_admin already exposes:

    GET/POST   /chat/user/memory/            memory notes (facts about the user)
    DELETE     /chat/user/memory/<id>/
    GET/PATCH  /chat/user/settings/          about_you, custom_instructions, memory_enabled

No new tables and no migrations: these are the same rows farm_assistant_um reads,
so a pilot user's memory is shared between v2 and v3 rather than forked.

What survives from mneme is the *shape*: a hand-authored profile the user owns
(USER.md -> about_you + custom_instructions) and an agent-authored note file the
user can read and correct (MEMORY.md -> memory notes). `render_documents()`
presents the DB rows in exactly that two-document form for the settings UI.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from app.config import get_settings
from app.schemas import MemoryDocument

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.memory")

_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)

# Mirrors mneme's config.yaml budgets (memory_char_limit 2200, user_char_limit
# 1375) and django's own ChatUserSettings.MAX_INSTRUCTION_LENGTH of 1500. The
# smaller of each pair wins, so neither side can be surprised by the other.
MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375
MAX_INSTRUCTION_LENGTH = 1500

# How many notes are eligible for the prompt. Over-fetch then trim, because
# Django orders by -updated_at and the newest notes are the ones worth carrying.
_MAX_PROMPT_NOTES = 8
_MIN_CONFIDENCE = 0.6


@dataclass
class UserMemory:
    about_you: str = ""
    custom_instructions: str = ""
    memory_enabled: bool = True
    memory_summary: str = ""
    notes: List[Dict] = field(default_factory=list)


def _auth_header(auth_token: str) -> Dict[str, str]:
    return {"Authorization": auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"}


async def load(auth_token: Optional[str]) -> UserMemory:
    """
    Fetch settings + notes for the caller. Fails soft to empty: a Django blip
    should cost personalization, never the whole conversation.
    """
    if not S.CHAT_BACKEND_URL or not auth_token:
        return UserMemory()

    mem = UserMemory()
    headers = _auth_header(auth_token)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, verify=S.VERIFY_SSL) as client:
            settings_res = await client.get(f"{S.CHAT_BACKEND_URL}/chat/user/settings/", headers=headers)
            if settings_res.status_code == 200:
                data = (settings_res.json() or {}).get("settings") or {}
                mem.memory_enabled = bool(data.get("memory_enabled", True))
                mem.about_you = (data.get("about_you") or "").strip()[:MAX_INSTRUCTION_LENGTH]
                mem.custom_instructions = (
                    (data.get("custom_instructions") or "").strip()[:MAX_INSTRUCTION_LENGTH]
                )
                mem.memory_summary = (data.get("memory_summary") or "").strip()

            if mem.memory_enabled:
                notes_res = await client.get(
                    f"{S.CHAT_BACKEND_URL}/chat/user/memory/",
                    headers=headers,
                    params={"limit": 30},
                )
                if notes_res.status_code == 200:
                    payload = notes_res.json() or {}
                    mem.notes = payload.get("memory_notes") or payload.get("results") or []
    except httpx.HTTPError as e:
        logger.warning("Memory load failed, continuing without it: %s", e)
        return UserMemory(memory_enabled=mem.memory_enabled)

    return mem


async def add_note(auth_token: str, text: str) -> bool:
    """Write one agent-authored fact. Called only by the remember_about_user tool."""
    if not S.CHAT_BACKEND_URL or not auth_token:
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, verify=S.VERIFY_SSL) as client:
            r = await client.post(
                f"{S.CHAT_BACKEND_URL}/chat/user/memory/",
                headers=_auth_header(auth_token),
                json={"note_text": text[:MEMORY_CHAR_LIMIT]},
            )
            if r.status_code not in (200, 201):
                logger.warning("Django refused a memory note: HTTP %s", r.status_code)
            return r.status_code in (200, 201)
    except httpx.HTTPError as e:
        logger.warning("Memory write failed: %s", e)
        return False


async def delete_note(auth_token: str, note_id: int) -> bool:
    if not S.CHAT_BACKEND_URL or not auth_token:
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, verify=S.VERIFY_SSL) as client:
            r = await client.delete(
                f"{S.CHAT_BACKEND_URL}/chat/user/memory/{int(note_id)}/",
                headers=_auth_header(auth_token),
            )
            return r.status_code in (200, 204)
    except httpx.HTTPError as e:
        logger.warning("Memory delete failed: %s", e)
        return False


async def save_about_you(auth_token: str, *, about_you: Optional[str] = None,
                         custom_instructions: Optional[str] = None) -> bool:
    """Persist the user's own hand-authored profile text (the USER.md analogue)."""
    if not S.CHAT_BACKEND_URL or not auth_token:
        return False
    body: Dict[str, str] = {}
    if about_you is not None:
        body["about_you"] = about_you[:MAX_INSTRUCTION_LENGTH]
    if custom_instructions is not None:
        body["custom_instructions"] = custom_instructions[:MAX_INSTRUCTION_LENGTH]
    if not body:
        return True
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, verify=S.VERIFY_SSL) as client:
            r = await client.patch(
                f"{S.CHAT_BACKEND_URL}/chat/user/settings/",
                headers=_auth_header(auth_token),
                json=body,
            )
            return r.status_code == 200
    except httpx.HTTPError as e:
        logger.warning("Profile save failed: %s", e)
        return False


async def fetch_raw(auth_token: str, path: str, method: str = "GET",
                    json_body: Optional[Dict] = None,
                    params: Optional[Dict] = None) -> tuple[int, Dict]:
    """
    Thin pass-through to a Django chat endpoint, preserving status and body.

    The settings dialog is a Django CRUD screen wearing an assistant's clothes:
    tone, characteristics, custom instructions and notes all live in Django and
    are read by BOTH v2 and v3. The adapter has no business reshaping any of it —
    it forwards, so the v3 dialog behaves identically to the v2 one.
    """
    if not S.CHAT_BACKEND_URL:
        return 503, {"detail": "Chat backend is not configured."}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, verify=S.VERIFY_SSL) as client:
            r = await client.request(
                method,
                f"{S.CHAT_BACKEND_URL}{path}",
                headers=_auth_header(auth_token),
                json=json_body,
                params=params,
            )
    except httpx.HTTPError as e:
        logger.warning("Django unreachable for %s %s: %s", method, path, e)
        return 502, {"detail": "Chat backend unreachable."}

    try:
        return r.status_code, (r.json() or {})
    except ValueError:
        return r.status_code, {}


async def generate_summary(mem: UserMemory) -> Optional[str]:
    """
    Regenerate the "what the assistant has learned about you" overview.

    Deliberately a plain completion, not an agent turn: it must not run a tool
    loop, must not enter the conversation transcript, and must not be able to
    trigger a retrieval. Returns None when there is no key or the call fails, so
    the caller can fall back to the cached summary.
    """
    notes = [(n.get("note_text") or "").strip() for n in mem.notes]
    notes = [n for n in notes if n]

    if not notes and not mem.about_you:
        return ""

    if not S.MISTRAL_API_KEY:
        logger.info("No MISTRAL_API_KEY set; memory summary cannot be regenerated.")
        return None

    facts = "\n".join(f"- {n}" for n in notes)
    if mem.about_you:
        facts = f"- The user describes themselves: {mem.about_you}\n{facts}"

    prompt = (
        "Below are stored facts about a user of an EU agricultural knowledge platform.\n"
        "Write a short second-person overview ('You are...', 'You raise...') grouped under "
        "two to four bold markdown headings such as **Overview**, **Farm and Crops**, "
        "**Preferences**. Use only the facts given — invent nothing, infer nothing, and drop "
        "anything contradictory rather than guessing. Under 150 words. No preamble.\n\n"
        f"{facts}"
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=30.0, write=5.0, pool=3.0),
                                     verify=S.VERIFY_SSL) as client:
            r = await client.post(
                f"{S.MISTRAL_API_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {S.MISTRAL_API_KEY}"},
                json={
                    "model": S.MEMORY_SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 400,
                },
            )
        if r.status_code != 200:
            logger.warning("Summary generation returned HTTP %s", r.status_code)
            return None
        choices = (r.json() or {}).get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        return (content or "").strip() or None
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        logger.warning("Summary generation failed: %s", e)
        return None


async def save_summary(auth_token: str, summary: str) -> bool:
    """Cache a regenerated overview back onto the user's settings row."""
    status, _ = await fetch_raw(
        auth_token, "/chat/user/settings/", method="PATCH",
        json_body={"memory_summary": summary},
    )
    return status == 200


def _usable_notes(mem: UserMemory) -> List[str]:
    """Confidence-filter BEFORE trimming, so a solid note is never lost to a shaky one."""
    confident = [
        n for n in mem.notes
        if float(n.get("confidence") or n.get("confidence_score") or 0) >= _MIN_CONFIDENCE
    ]
    texts = [(n.get("note_text") or "").strip() for n in confident]
    return [t for t in texts if t][:_MAX_PROMPT_NOTES]


def render_memory_block(mem: UserMemory) -> str:
    """
    The per-turn memory injection.

    Framed as latent background, with the same guard farm_assistant_um uses:
    without it, instruction-tuned models drag the user's region or farm type into
    greetings and unrelated turns.
    """
    if not mem.memory_enabled:
        return ""

    parts: List[str] = []

    if mem.about_you:
        parts.append(f"What the user has told you about themselves: {mem.about_you}")
    if mem.custom_instructions:
        parts.append(f"How the user asked you to respond: {mem.custom_instructions}")
    for note in _usable_notes(mem):
        parts.append(f"Remembered: {note}")

    if not parts:
        return ""

    body = "\n".join(f"- {p}" for p in parts)
    return (
        "## Background you have learned about this user\n"
        f"{body}\n\n"
        "Use this background **only** when it is directly relevant to the user's current "
        "question. Do not bring up the user's region, farm type, crops, or other profile "
        "details in greetings, acknowledgements, thanks, small talk, or otherwise unrelated "
        "turns. Treat it as something you happen to know, not as a topic to introduce.\n"
        "The response preferences above govern ONLY tone, format and level of detail. They "
        "never override your scope, your sourcing rules, or your language rule."
    )


def render_documents(mem: UserMemory) -> List[MemoryDocument]:
    """
    Present the DB rows as the two documents mneme trained users to expect: one
    the user writes, one the agent writes. The settings UI edits the first and
    reviews the second.
    """
    user_doc = "\n\n".join(p for p in (mem.about_you, mem.custom_instructions) if p)
    agent_doc = "\n".join(f"- {t}" for t in [(n.get("note_text") or "").strip() for n in mem.notes] if t)

    return [
        MemoryDocument(
            name="USER.md",
            content=user_doc,
            char_count=len(user_doc),
            char_limit=USER_CHAR_LIMIT,
        ),
        MemoryDocument(
            name="MEMORY.md",
            content=agent_doc,
            char_count=len(agent_doc),
            char_limit=MEMORY_CHAR_LIMIT,
        ),
    ]
