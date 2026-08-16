# app/services/suggestion_service.py
"""
Opening prompts for an empty chat, derived from what the agent remembers.

Not the same thing as the static suggestion list a generic chat UI ships (three
hardcoded strings about study tips and CSS). These are generated per user from
their stored profile and memory notes, so someone who farms dairy in Brittany
opens the page and is offered Brittany dairy questions. That is the whole thesis
of v3 rendered in one screen, and it costs one cheap completion per day.

Like the memory summary, this is a PLAIN completion, never an agent turn: it
must not run a tool loop, must not enter the conversation transcript, and must
not be able to trigger a retrieval.

Falls back to a fixed EU-FarmBook set whenever there is nothing remembered, no
provider key, or the model misbehaves — an empty chat should never look broken.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx

from app.config import get_settings
from app.services import memory_service

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.suggestions")

# Generation is per user and rarely changes; a page reload must not cost a call.
_CACHE_TTL_SECONDS = 3600.0
_MAX_CACHED_USERS = 5000
_cache: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}


@dataclass
class Suggestion:
    title: str      # the bold line — a short imperative
    subtitle: str   # the qualifier under it
    prompt: str     # what is actually sent when clicked


# Deliberately generic and platform-shaped: shown to a user the agent knows
# nothing about yet, so they must make sense to any EU-FarmBook visitor.
_DEFAULTS: List[Dict[str, str]] = [
    {
        "title": "Explain cover crops",
        "subtitle": "benefits and how to choose one",
        "prompt": "What does EU-FarmBook say about choosing cover crops?",
    },
    {
        "title": "Reduce input costs",
        "subtitle": "practical measures from EU projects",
        "prompt": "Which practices in EU-FarmBook help reduce fertiliser and input costs?",
    },
    {
        "title": "Find research on soil health",
        "subtitle": "what the platform holds",
        "prompt": "What material does EU-FarmBook have on improving soil health?",
    },
]

_PROMPT = """You write opening suggestions for an agricultural assistant on the
EU-FarmBook platform, for one specific user.

Here is what is known about them:
{profile}

Write exactly 3 suggestions they would plausibly want to ask, as JSON:
[{{"title": "...", "subtitle": "...", "prompt": "..."}}]

Rules:
- Every suggestion must be about agriculture, farming, food systems, or
  EU-FarmBook itself. Nothing else, whatever the profile says.
- Ground them in the specifics above — their region, crops, livestock, role.
  A suggestion that would suit any farmer is a wasted suggestion.
- "title" is 2-5 words, imperative. "subtitle" is a short qualifier, under 8
  words, lower case. "prompt" is the full question, one sentence.
- Do not invent facts about the user that are not listed above.
- Reply with the JSON array and nothing else.
"""


def _profile_lines(mem: memory_service.UserMemory) -> str:
    parts: List[str] = []
    if mem.about_you:
        parts.append(f"- They describe themselves: {mem.about_you}")
    for note in mem.notes:
        text = (note.get("note_text") or "").strip()
        if text:
            parts.append(f"- {text}")
    return "\n".join(parts[:12])


def _cache_get(user_uuid: str) -> Optional[List[Dict[str, str]]]:
    entry = _cache.get(user_uuid)
    if not entry:
        return None
    stamped, suggestions = entry
    if time.monotonic() - stamped > _CACHE_TTL_SECONDS:
        _cache.pop(user_uuid, None)
        return None
    return suggestions


def _cache_put(user_uuid: str, suggestions: List[Dict[str, str]]) -> None:
    if len(_cache) >= _MAX_CACHED_USERS:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    _cache[user_uuid] = (time.monotonic(), suggestions)


def _parse(content: str) -> List[Dict[str, str]]:
    """
    Pull the JSON array out of a model reply, tolerating a code fence.

    Anything malformed returns empty and the caller falls back — a broken
    suggestion list is not worth an error on an otherwise working page.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.removeprefix("json").strip()

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []

    out: List[Dict[str, str]] = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not title or not prompt:
            continue
        out.append({
            "title": title[:60],
            "subtitle": str(item.get("subtitle") or "").strip()[:80],
            "prompt": prompt[:300],
        })
    return out[:3]


async def get_suggestions(auth_token: str, user_uuid: str) -> Tuple[List[Dict[str, str]], bool]:
    """
    Return (suggestions, personalised). `personalised` is False when these are
    the platform defaults, so the UI can label them honestly if it wants to.
    """
    cached = _cache_get(user_uuid)
    if cached is not None:
        return cached, True

    mem = await memory_service.load(auth_token)
    profile = _profile_lines(mem) if mem.memory_enabled else ""

    # Nothing remembered yet, or the user turned memory off: defaults, and do
    # not spend a completion discovering that.
    if not profile or not S.MISTRAL_API_KEY:
        return _DEFAULTS, False

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=20.0, write=5.0, pool=3.0),
            verify=S.VERIFY_SSL,
        ) as client:
            r = await client.post(
                f"{S.MISTRAL_API_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {S.MISTRAL_API_KEY}"},
                json={
                    "model": S.MEMORY_SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": _PROMPT.format(profile=profile)}],
                    "temperature": 0.4,
                    "max_tokens": 400,
                },
            )
        if r.status_code != 200:
            logger.warning("Suggestion generation returned HTTP %s", r.status_code)
            return _DEFAULTS, False

        choices = (r.json() or {}).get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else ""
        suggestions = _parse(content or "")
    except (httpx.HTTPError, ValueError, IndexError, AttributeError) as e:
        logger.warning("Suggestion generation failed: %s", e)
        return _DEFAULTS, False

    if not suggestions:
        return _DEFAULTS, False

    _cache_put(user_uuid, suggestions)
    return suggestions, True


def invalidate(user_uuid: str) -> None:
    """Drop a user's cached suggestions — call when their memory changes."""
    _cache.pop(user_uuid, None)


def reset() -> None:
    """Test helper."""
    _cache.clear()
