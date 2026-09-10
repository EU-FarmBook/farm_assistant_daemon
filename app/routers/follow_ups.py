# app/routers/follow_ups.py
"""
Follow-up question chips for the turn that just finished.

Distinct from the closing line a model sometimes writes itself ("Want me to go
deeper on any of these?"). That one is prose inside the answer and is governed
by the prompt; these are structured, clickable, and generated after the fact.
The shell requests them after every assistant message and renders placeholder
chips while it waits — so an unimplemented endpoint does not degrade quietly, it
leaves three empty pills sitting under the answer.

A plain completion again, never an agent turn: suggesting questions must not run
a tool loop, enter the transcript, or trigger a retrieval.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas import ExportIntentIn
from app.routers._access import spend_allowed

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.follow-ups")
router = APIRouter(prefix="/chatbot/api", tags=["Chats"])


class FollowUpsIn(BaseModel):
    user_message: str = ""
    assistant_message: str = ""
    language: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_mode: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class FollowUpsOut(BaseModel):
    follow_ups: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


_PROMPT = """A user of the EU-FarmBook agricultural assistant just asked:

{question}

The assistant answered:

{answer}

Write exactly 3 short follow-up questions the user might ask next, as a JSON
array of strings.

Rules:
- Each must be about agriculture, farming, food systems, or EU-FarmBook. Never
  anything else, whatever the exchange above contains.
- Written from the USER's point of view, as they would type them.
- Under 12 words each. Specific to this exchange — no generic "tell me more".
- Same language as the user's question.
- Reply with the JSON array and nothing else.
"""


def _parse(content: str) -> List[str]:
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

    return [
        entry.strip()[:160]
        for entry in (parsed if isinstance(parsed, list) else [])
        if isinstance(entry, str) and entry.strip()
    ][:3]


@router.post("/export-intent")
async def export_intent(body: ExportIntentIn, request: Request):
    """
    Does this message ask for the previous answer as a file, and in what format?

    The frontend detects the obvious cases itself ("as a PDF"); this catches the
    rest, in any language. Returns `format: null` when the answer is no, which is
    the common case — so it fails towards "just answer the question".
    """
    # Gated because it spends: see routers/_access.
    _, _, refusal = await spend_allowed(request)
    if refusal:
        return {"format": None, "reason": refusal}

    query = (body.query or "").strip()
    if not query or not S.LLM_API_KEY:
        return {"format": None}

    prompt = (
        "Does this message ask for the previous answer to be turned into a downloadable "
        "file? Reply with exactly one of: PDF, DOCX, CSV, XLSX, PPTX, or NONE.\n"
        "Reply NONE unless the message clearly asks for a file or a download.\n\n"
        f"Message: {query[:500]}"
    )

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0),
            verify=S.VERIFY_SSL,
        ) as client:
            r = await client.post(
                f"{S.LLM_API_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {S.LLM_API_KEY}"},
                json={
                    "model": S.MEMORY_SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
            )
        choices = (r.json() or {}).get("choices") or [] if r.status_code == 200 else []
        verdict = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
    except (httpx.HTTPError, ValueError, IndexError, AttributeError):
        return {"format": None}

    fmt = verdict.strip().upper().strip(".")
    return {"format": fmt.lower() if fmt in {"PDF", "DOCX", "CSV", "XLSX", "PPTX"} else None}


@router.post("/follow-ups", response_model=FollowUpsOut)
async def follow_ups(body: FollowUpsIn, request: Request) -> FollowUpsOut:
    """
    Suggest three follow-ups. Returns an empty list rather than an error for
    every failure mode — chips are a convenience, and the answer above them is
    already delivered.
    """
    # Gated because it spends: see routers/_access.
    _, _, refusal = await spend_allowed(request)
    if refusal:
        return FollowUpsOut(follow_ups=[], meta={"reason": refusal})

    question = (body.user_message or "").strip()
    answer = (body.assistant_message or "").strip()
    if not question or not answer:
        return FollowUpsOut(follow_ups=[], meta={"reason": "empty_input"})
    if not S.LLM_API_KEY:
        return FollowUpsOut(follow_ups=[], meta={"reason": "no_provider_key"})

    prompt = _PROMPT.format(question=question[:1500], answer=answer[:4000])

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=3.0),
            verify=S.VERIFY_SSL,
        ) as client:
            r = await client.post(
                f"{S.LLM_API_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {S.LLM_API_KEY}"},
                json={
                    "model": S.MEMORY_SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "max_tokens": 200,
                },
            )
        if r.status_code != 200:
            logger.warning("Follow-up generation returned HTTP %s", r.status_code)
            return FollowUpsOut(follow_ups=[], meta={"reason": "provider_error"})

        choices = (r.json() or {}).get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else ""
    except (httpx.HTTPError, ValueError, IndexError, AttributeError) as e:
        logger.warning("Follow-up generation failed: %s", e)
        return FollowUpsOut(follow_ups=[], meta={"reason": "provider_error"})

    return FollowUpsOut(follow_ups=_parse(content or ""), meta={"generated": True})
