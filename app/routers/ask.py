# app/routers/ask.py
"""
The streaming chat endpoint, speaking farm_assistant_um's SSE contract.

Event vocabulary is fixed by the v2 UI and must not drift: `status`, `sources`,
`grounding`, `token`, `final`, `timing`, `done`, `app_error`. The v3 shell is a
copy of the v2 shell, so anything new here has to be added to both or it is
simply not rendered.

What differs from v2 is what happens between `status` and the first `token`: v2
retrieves, then generates once. v3 hands the question to an agent that decides
whether and how often to retrieve. That is the thing the pilot is measuring, so
the route deliberately does not second-guess it — no pre-retrieval, no routing,
no rewrite.
"""

import json
import logging
import time
from typing import AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.services import memory_service, tool_server
from app.services.auth_service import decode_token_email, resolve_user_uuid
from app.services.hermes_client import HermesUnavailable, stream_chat
from app.services.profile_registry import ProfileNotProvisioned, resolve_profile
from app.services.scope import system_prompt

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.ask")
router = APIRouter()

# Replayed history budget. Hermes carries its own transcript per session id, so
# this is belt-and-braces for a fresh session id on an existing platform chat —
# not the primary continuity mechanism.
_MAX_HISTORY_MESSAGES = 20
_MAX_HISTORY_CHARS = 12000


def _parse_client_history(raw: Optional[str]) -> List[Dict[str, str]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    out: List[Dict[str, str]] = []
    budget = _MAX_HISTORY_CHARS
    for item in parsed[-_MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        budget -= len(content)
        if budget <= 0:
            break
        out.append({"role": role, "content": content})
    return out


@router.get("/chatbot/api/chats/message/stream", tags=["Chats"])
@router.get("/chatbot/api/chats/{session_id}/message/stream", tags=["Chats"])
async def stream_message(
    request: Request,
    session_id: Optional[str] = None,
    q: str = Query(..., min_length=1),
    page: int = Query(1),
    max_tokens: int = Query(-1),
    pause_personalization: bool = Query(False),
    replace_history: bool = Query(False),
    client_history: Optional[str] = Query(None),
):
    auth_token = request.headers.get("Authorization", "")
    user_uuid = await resolve_user_uuid(auth_token) if auth_token else None

    if S.REQUIRE_CHAT_AUTH and not user_uuid:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        # Safe to read the email claim here and not before: resolve_user_uuid()
        # has verified the token's signature, so its claims are the issuer's.
        profile = resolve_profile(user_uuid, email=decode_token_email(auth_token))
    except ProfileNotProvisioned:
        # Deliberately explicit rather than a generic 403: everyone outside the
        # pilot will hit this, and "you are not in the pilot" is the useful thing
        # for the frontend to show.
        raise HTTPException(
            status_code=403,
            detail="This experimental assistant is limited to the pilot group.",
        )

    started = time.monotonic()

    async def emit(event: str, data) -> Dict[str, str]:
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        return {"event": event, "data": data}

    async def gen() -> AsyncIterator[Dict[str, str]]:
        tool_server.begin_turn(profile, auth_token=auth_token, user_uuid=user_uuid)
        # Version of the citation register we have already pushed to the client.
        # 0 = nothing sent yet. A latch ("sent / not sent") would drop the second
        # search's sources on a multi-hop turn, leaving the UI showing hop 1
        # while the answer cites hop 2.
        sent_version = 0
        answer_parts: List[str] = []

        try:
            yield await emit("status", {"stage": "thinking"})

            memory_block = ""
            if not pause_personalization:
                mem = await memory_service.load(auth_token)
                memory_block = memory_service.render_memory_block(mem)

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt(memory_block or None)}
            ]
            if replace_history:
                messages.extend(_parse_client_history(client_history))
            messages.append({"role": "user", "content": q})

            async for delta in stream_chat(
                profile=profile,
                user_uuid=user_uuid,
                session_id=session_id,
                messages=messages,
            ):
                # Citations as soon as the agent has retrieved, not after the
                # answer: the UI renders the source rail alongside the text. The
                # payload is always the CUMULATIVE register, so a re-emit
                # replaces the list rather than needing the client to merge.
                parked = tool_server.peek_sources(profile)
                if parked is not None and parked[0] > sent_version:
                    version, sources = parked
                    first_emit = sent_version == 0
                    sent_version = version
                    yield await emit("sources", [s.model_dump() for s in sources])
                    if first_emit:
                        yield await emit(
                            "grounding",
                            {"mode": "euf_supported" if sources else "general_fallback"},
                        )

                answer_parts.append(delta)
                yield await emit("token", {"text": delta})

            answer = "".join(answer_parts)

            # A final sweep: a retrieval that landed after the last token (or a
            # turn with no tokens at all) would otherwise never be published.
            parked = tool_server.peek_sources(profile)
            if parked is not None and parked[0] > sent_version:
                version, sources = parked
                first_emit = sent_version == 0
                sent_version = version
                yield await emit("sources", [s.model_dump() for s in sources])
                if first_emit:
                    yield await emit(
                        "grounding",
                        {"mode": "euf_supported" if sources else "general_fallback"},
                    )

            if sent_version == 0:
                # The agent answered without retrieving — a refusal, a greeting,
                # or a scope decline. Say so honestly rather than implying
                # platform grounding that did not happen.
                yield await emit("sources", [])
                yield await emit("grounding", {"mode": "general_fallback"})

            yield await emit("final", {"answer": answer})
            yield await emit("timing", {"total_ms": int((time.monotonic() - started) * 1000)})
            yield await emit("done", {"ok": True})

        except HermesUnavailable as e:
            logger.error("Turn failed for profile=%s: %s", profile, e)
            yield await emit(
                "app_error",
                {"message": "The assistant is temporarily unavailable. Please try again."},
            )
        except Exception:
            logger.exception("Unhandled error in turn for profile=%s", profile)
            yield await emit(
                "app_error",
                {"message": "Something went wrong while answering. Please try again."},
            )
        finally:
            ctx = tool_server.end_turn(profile)
            if ctx and ctx.remembered:
                logger.info("profile=%s stored %d memory note(s)", profile, len(ctx.remembered))

    return EventSourceResponse(gen())
