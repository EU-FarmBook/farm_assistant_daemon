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
import re
import time
from typing import AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.services import attachment_service, memory_service, rate_limit, tool_server
from app.services.auth_service import (
    decode_token_email,
    decode_token_first_name,
    resolve_user_uuid,
)
from app.services.hermes_client import HermesUnavailable, complete_chat, stream_chat
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


# A message this short is a greeting, a "thanks", or a one-word aside in any
# language. Below it, pre-retrieval is deferred — the agent can still search if
# it turns out to matter. Length, not keywords: a word list would be brittle and
# English-only, which is exactly why v2's were removed.
_PREFETCH_MIN_CHARS = 15


def _search_query(question: str, history: List[Dict[str, str]]) -> str:
    """
    The query to pre-retrieve with.

    A follow-up like "and for maize?" is meaningless to a retriever on its own,
    so the previous user turn is prepended when the question looks like it leans
    on context. v2 spends an LLM call resolving this properly; here the agent
    can re-query if the cheap heuristic guesses wrong, so a heuristic is the
    right trade.
    """
    question = question.strip()
    previous = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"),
        "",
    )
    if not previous:
        return question
    if len(question) <= 60:
        return f"{previous.strip()} {question}"[:500]
    return question


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
    doc_ids: Optional[str] = Query(None),
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

    # Before any model call: an agent turn is several billed calls, so a
    # refusal has to happen here rather than after the spend.
    try:
        rate_limit.check_and_record(user_uuid)
    except rate_limit.RateLimited as limited:
        raise HTTPException(
            status_code=429,
            detail=(
                "You have reached the limit for now. Please try again in a "
                f"{'few moments' if limited.scope == 'minute' else 'while'}."
            ),
            headers={"Retry-After": str(limited.retry_after_seconds)},
        )

    started = time.monotonic()

    async def emit(event: str, data) -> Dict[str, str]:
        """
        Frame one SSE event.

        Strings are sent verbatim; everything else is JSON. That distinction is
        the contract, not a convenience: the shell appends `token` payloads to
        the answer WITHOUT parsing them, so a JSON-wrapped token renders as
        literal `{"text": "Bon"}` in the chat. Only `token` is a bare string.
        """
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        return {"event": event, "data": data}

    async def gen() -> AsyncIterator[Dict[str, str]]:
        tool_server.begin_turn(
            profile, auth_token=auth_token, user_uuid=user_uuid, user_message=q,
        )
        # Version of the citation register we have already pushed to the client.
        # 0 = nothing sent yet. A latch ("sent / not sent") would drop the second
        # search's sources on a multi-hop turn, leaving the UI showing hop 1
        # while the answer cites hop 2.
        sent_version = 0
        answer_parts: List[str] = []

        try:
            yield await emit("status", {"stage": "agent", "message": "Working on your question..."})

            memory_block = ""
            if not pause_personalization:
                mem = await memory_service.load(auth_token)
                memory_block = memory_service.render_memory_block(
                    mem, first_name=decode_token_first_name(auth_token)
                )
                # Map [M1], [M2]... to real note ids so forget_about_user can
                # act on what the agent sees.
                tool_server.set_note_ids(
                    profile,
                    [n.get("id") for n in memory_service.usable_notes(mem) if n.get("id")],
                )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt(memory_block or None)}
            ]
            history = _parse_client_history(client_history) if replace_history else []
            if history:
                messages.extend(history)

            # --- Retrieve first, then let the agent search again -------------
            #
            # v2 retrieves before generating and therefore cannot answer
            # ungrounded; a pure agent decides for itself and sometimes does not
            # look at all — which is how an answer came to assert EU-FarmBook had
            # nothing on pig manure without a single search. This restores v2's
            # floor: every substantive turn starts with real passages.
            #
            # It goes through the SAME tool the agent calls, so passages land in
            # one citation register and a later agent hop continues the numbering
            # instead of restarting it.
            sources_block = ""
            if len(q.strip()) >= _PREFETCH_MIN_CHARS:
                yield await emit(
                    "status", {"stage": "search", "message": "Searching EU-FarmBook..."}
                )
                prefetch = await tool_server.search_eu_farmbook(
                    _search_query(q, history), profile=profile
                )
                passages = prefetch.get("passages") or []
                if passages:
                    numbered = "\n\n".join(f"[{p['n']}] {p['text']}" for p in passages)
                    quality = (prefetch.get("quality") or {}).get("verdict", "unknown")
                    sources_block = (
                        "EU-FarmBook passages retrieved for this question:\n\n"
                        f"{numbered}\n\n"
                        f"(Relevance of this set: {quality}.)"
                        + (
                            " These look like a poor match — search again with more "
                            "specific terms before answering, and say so plainly if it "
                            "stays weak."
                            if quality == "weak" else ""
                        )
                        + "\nCite what you use by these numbers. Search again with "
                        "search_eu_farmbook if they do not cover the question.\n\n"
                    )
                elif not prefetch.get("ok", True):
                    # Search is down. Say so, and forbid the inference the model
                    # would otherwise make from an empty result.
                    logger.error("Pre-retrieval failed for profile=%s", profile)
                    sources_block = (
                        f"{prefetch.get('error')}\n\n"
                    )
                else:
                    sources_block = (
                        "A search of EU-FarmBook for this question returned nothing. "
                        "Try one more search with different terms before concluding "
                        "the platform has no material on it.\n\n"
                    )
            # Attached documents ride with the question as user-provided
            # material, explicitly not as platform sources — the agent cites
            # EU-FarmBook by number, and an uploaded file must never be
            # presented as though it came from the platform.
            question = q
            if doc_ids:
                attached = attachment_service.build_context(
                    [d for d in doc_ids.split(",") if d.strip()], user_uuid
                )
                if attached:
                    question = f"{attached}\n\n{q}"

            messages.append({"role": "user", "content": f"{sources_block}{question}"})

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
                yield await emit("token", delta)

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

            if not answer.strip():
                # The stream said nothing. Ask again without streaming before
                # giving up: Hermes drops content on the streaming path in some
                # states, and an answer delivered in one chunk beats none.
                logger.warning(
                    "Empty stream for profile=%s — retrying without streaming", profile
                )
                answer = await complete_chat(
                    profile=profile,
                    user_uuid=user_uuid,
                    session_id=session_id,
                    messages=messages,
                )
                if answer.strip():
                    yield await emit("token", answer)

            if not answer.strip():
                # An agent that completes with no text is a failure wearing a
                # success: the UI renders an empty bubble and Django then 400s
                # the turn log (it requires both messages non-empty). It is how
                # "Unknown provider" surfaced — as silence. Say so instead.
                logger.error(
                    "Empty completion for profile=%s — check the agent's provider config",
                    profile,
                )
                yield await emit(
                    "app_error",
                    {"message": "The assistant returned an empty answer. Please try again."},
                )
                return

            # Show sources only if the answer actually cited them.
            #
            # Pre-retrieval runs on every substantive turn, so a question like
            # "who am I?" pulled five unrelated documents and the UI labelled
            # the reply "Grounded in EU-FarmBook" while it cited nothing. That
            # is worse than showing no sources: it dresses an ungrounded answer
            # in the authority of the platform.
            if sent_version and not re.search(r"\[\d+\]", answer):
                logger.info(
                    "Answer for profile=%s cited nothing — clearing the source rail", profile
                )
                yield await emit("sources", [])
                yield await emit("grounding", {"mode": "general_fallback"})

            yield await emit("final", {"text": answer})
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
