# app/routers/ask.py
"""
The streaming chat endpoint, speaking farm_assistant_um's SSE contract.

Event vocabulary is fixed by the v2 UI and must not drift: `status`, `sources`,
`grounding`, `token`, `final`, `timing`, `done`, `app_error`. The v3 shell is a
copy of the v2 shell, so anything new here has to be added to both or it is
simply not rendered.

EVERY terminal path ends with `done` — `{"ok": true}` on success, or
`{"ok": false, "code": ...}` after an `app_error`. That is not cosmetic: the
failure paths used to end the generator with `app_error` and nothing else, and a
clean EOF is what the EventSource spec tells a client to RECONNECT on. So each
failure silently re-ran a whole billed agent turn, on a loop, for as long as the
underlying fault lasted. `app_error` also carries a stable `code`, so a client
can branch without parsing an English sentence.

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
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
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

# Below this a caller-supplied max_tokens is refused rather than honoured. The
# agent model is a reasoning model whose thinking is billed against the same
# budget, so a small cap produces an empty answer — the failure would look like
# the assistant breaking, not like the cap doing its job.
_MIN_USEFUL_MAX_TOKENS = 512


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


async def _claim_turn(request: Request, q: str) -> Tuple[str, str, str]:
    """
    Authenticate, gate, meter and CLAIM the turn. Returns (auth_token, uuid, profile).

    Every refusal here is a real HTTP status, before a byte of the response body
    exists — 401 no token, 403 outside the pilot, 503 our own provisioning
    failure, 429 rate limited, 409 a turn already in flight. Both doors call
    this, so neither can drift from the other on who is allowed to spend.
    """
    auth_token = request.headers.get("Authorization", "")
    user_uuid = await resolve_user_uuid(auth_token) if auth_token else None

    if S.REQUIRE_CHAT_AUTH and not user_uuid:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        # Safe to read the email claim here and not before: resolve_user_uuid()
        # has verified the token's signature, so its claims are the issuer's.
        profile = resolve_profile(user_uuid, email=decode_token_email(auth_token))
    except ProfileNotProvisioned as e:
        # Two different failures used to share one message. Being outside the
        # pilot is a 403 and true for most callers; a volume that is not mounted,
        # a missing template or an unset HERMES_MODEL is OUR fault, and telling
        # that user "you are not in the pilot" sends the operator hunting the
        # roster instead of the deployment.
        if "provisioning failed" in (e.reason or "") or "invalid profile" in (e.reason or ""):
            logger.error("Provisioning failure for uuid=%s: %s", user_uuid, e.reason)
            raise HTTPException(
                status_code=503,
                detail="The assistant could not be prepared for your account. Please try again.",
            )
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

    # Claim the turn here, not inside the generator: a refusal is then a real
    # HTTP status the client can branch on, like the 401/403/429 above, instead
    # of an SSE event arriving after a 200. The generator's finally still ends
    # the turn, including on client disconnect.
    try:
        tool_server.begin_turn(
            profile, auth_token=auth_token, user_uuid=user_uuid, user_message=q,
        )
    except tool_server.TurnInProgress:
        raise HTTPException(
            status_code=409,
            detail=(
                "You already have a question being answered. Wait for it to finish "
                "before sending another."
            ),
        )
    return auth_token, user_uuid, profile


_STREAM_DESCRIPTION = """
Answer one question as a Server-Sent Events stream.

**Events, in order:** `status` (one or more), `sources`, `grounding`, `token`
(many), `final`, `timing`, `done`. Any turn may instead end `app_error` then
`done`.

**`token` is a BARE STRING**, not JSON — every other event's `data` is JSON. A
client that `JSON.parse`s every event will mangle answers containing
JSON-shaped text.

**Reassemble `data:` lines per the SSE spec.** A token containing newlines is
framed as several `data:` lines (that is the spec, not a quirk); the correct
reading is to join them with `\n` and drop one trailing newline. A reader that
treats each `data:` line as its own token silently flattens every markdown
answer — tables, lists and paragraph breaks all collapse.

**Other wire details:** separators are CRLF; a comment-only keepalive frame
(`: ping - <timestamp>`) arrives about every 15s and must be skipped; the whole
answer is delivered twice, streamed as `token` and complete in `final.text`, so
render one or the other, not both.

**`sources` REPLACES, never appends.** It is re-emitted as later retrieval hops
land, always as the cumulative list, and an answer that cites nothing ends with
an empty `sources` — merging them instead of replacing keeps a rail the answer
does not support.

**`done` always ends the stream:** `{"ok": true}`, or `{"ok": false, "code":
...}` after an `app_error`. Do not reconnect on it — a reconnect re-runs a
billed agent turn.

**Pre-stream failures are HTTP statuses, not events:** 401 no token, 403 outside
the pilot, 409 a turn already in flight for this user, 429 rate limited (with
`Retry-After`).
"""


@router.get(
    "/chatbot/api/chats/message/stream",
    tags=["Chats"],
    description=_STREAM_DESCRIPTION,
)
@router.get(
    "/chatbot/api/chats/{session_id}/message/stream",
    tags=["Chats"],
    description=_STREAM_DESCRIPTION,
)
async def stream_message(
    request: Request,
    session_id: Optional[str] = None,
    q: str = Query(..., min_length=1),
    max_tokens: int = Query(
        -1,
        description=(
            "Cap the agent's completion, reasoning tokens included. -1 leaves the "
            "profile's own max_tokens in force. Values under "
            f"{_MIN_USEFUL_MAX_TOKENS} are ignored: a reasoning model can spend "
            "that entirely on thinking and return no answer at all."
        ),
    ),
    pause_personalization: bool = Query(
        False, description="Skip the user's remembered profile for this turn."
    ),
    replace_history: bool = Query(
        False,
        description=(
            "Prepend `client_history` to this turn. It does NOT replace the agent's "
            "own per-session transcript, which Hermes keeps against the session id; "
            "the two are additive. Without this flag `client_history` is ignored."
        ),
    ),
    client_history: Optional[str] = Query(
        None, description="JSON array of {role, content}; only read when replace_history=true."
    ),
    doc_ids: Optional[str] = Query(
        None, description="Comma-separated ids from POST /chatbot/api/files/document."
    ),
):
    auth_token, user_uuid, profile = await _claim_turn(request, q)

    started = time.monotonic()

    async def sse() -> AsyncIterator[Dict[str, str]]:
        """Frame each pair. `token` stays a bare string; everything else is JSON."""
        async for event, data in _turn_events(
            profile=profile,
            auth_token=auth_token,
            user_uuid=user_uuid,
            session_id=session_id,
            q=q,
            max_tokens=max_tokens,
            pause_personalization=pause_personalization,
            replace_history=replace_history,
            client_history=client_history,
            doc_ids=doc_ids,
            started=started,
        ):
            payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            yield {"event": event, "data": payload}

    return EventSourceResponse(sse())


async def _turn_events(
    *,
    profile: str,
    auth_token: str,
    user_uuid: str,
    session_id: Optional[str],
    q: str,
    max_tokens: int,
    pause_personalization: bool,
    replace_history: bool,
    client_history: Optional[str],
    doc_ids: Optional[str],
    started: float,
) -> AsyncIterator[Tuple[str, Any]]:
    """
    One turn, as a sequence of (event, data) pairs. THE implementation.

    Both doors consume this: the SSE route frames each pair as an event, and the
    non-streaming route drains it and assembles one JSON body. Keeping it single
    means the turn claim, the citation register, the memory block, the gates and
    the rate limit happen exactly once and identically either way — a second
    copy of this would be a second set of bugs.

    The caller must already have claimed the turn (see _claim_turn); this
    generator owns ENDING it, including on client disconnect.
    """
    # The turn was already claimed above; this generator owns ending it.
    # Version of the citation register we have already pushed to the client.
    # 0 = nothing sent yet. A latch ("sent / not sent") would drop the second
    # search's sources on a multi-hop turn, leaving the UI showing hop 1
    # while the answer cites hop 2.
    sent_version = 0
    answer_parts: List[str] = []

    try:
        yield ("status", {"stage": "agent", "message": "Working on your question..."})

        memory_block = ""
        if not pause_personalization:
            mem = await memory_service.load(auth_token)
            memory_block = memory_service.render_memory_block(
                mem, first_name=decode_token_first_name(auth_token)
            )
            # Map [M1], [M2]... to real note ids so forget_about_user can
            # act on what the agent sees.
            tool_server.set_notes(profile, memory_service.usable_notes(mem))

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt(memory_block or None)}
        ]
        history = _parse_client_history(client_history) if replace_history else []
        if client_history and not replace_history:
            logger.info(
                "client_history supplied without replace_history=true for profile=%s; "
                "ignoring it", profile,
            )
        elif client_history and not history:
            logger.info(
                "client_history for profile=%s parsed to nothing (malformed, or no "
                "usable user/assistant turns)", profile,
            )
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
            yield (
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

        requested_cap = max_tokens if max_tokens and max_tokens > 0 else None
        if requested_cap and requested_cap < _MIN_USEFUL_MAX_TOKENS:
            logger.info(
                "Ignoring max_tokens=%s for profile=%s: below the %s needed for a "
                "reasoning model to produce any answer.",
                requested_cap, profile, _MIN_USEFUL_MAX_TOKENS,
            )
            requested_cap = None

        async for delta in stream_chat(
            profile=profile,
            user_uuid=user_uuid,
            session_id=session_id,
            messages=messages,
            max_tokens=requested_cap,
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
                yield ("sources", [s.model_dump() for s in sources])
                if first_emit:
                    yield (
                        "grounding",
                        {"mode": "euf_supported" if sources else "general_fallback"},
                    )

            answer_parts.append(delta)
            yield ("token", delta)

        answer = "".join(answer_parts)

        # A final sweep: a retrieval that landed after the last token (or a
        # turn with no tokens at all) would otherwise never be published.
        parked = tool_server.peek_sources(profile)
        if parked is not None and parked[0] > sent_version:
            version, sources = parked
            first_emit = sent_version == 0
            sent_version = version
            yield ("sources", [s.model_dump() for s in sources])
            if first_emit:
                yield (
                    "grounding",
                    {"mode": "euf_supported" if sources else "general_fallback"},
                )

        if sent_version == 0:
            # The agent answered without retrieving — a refusal, a greeting,
            # or a scope decline. Say so honestly rather than implying
            # platform grounding that did not happen.
            yield ("sources", [])
            yield ("grounding", {"mode": "general_fallback"})

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
                yield ("token", answer)

        if not answer.strip():
            # An agent that completes with no text is a failure wearing a
            # success: the UI renders an empty bubble and Django then 400s
            # the turn log (it requires both messages non-empty). It is how
            # "Unknown provider" surfaced — as silence. Say so instead.
            logger.error(
                "Empty completion for profile=%s — check the agent's provider config",
                profile,
            )
            yield (
                "app_error",
                {
                    "message": "The assistant returned an empty answer. Please try again.",
                    "code": "empty_answer",
                },
            )
            yield ("done", {"ok": False, "code": "empty_answer"})
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
            yield ("sources", [])
            yield ("grounding", {"mode": "general_fallback"})

        yield ("final", {"text": answer})
        yield ("timing", {"total_ms": int((time.monotonic() - started) * 1000)})
        yield ("done", {"ok": True})

    except HermesUnavailable as e:
        logger.error("Turn failed for profile=%s: %s", profile, e)
        yield (
            "app_error",
            {
                "message": "The assistant is temporarily unavailable. Please try again.",
                "code": "agent_unavailable",
            },
        )
        yield ("done", {"ok": False, "code": "agent_unavailable"})
    except Exception:
        logger.exception("Unhandled error in turn for profile=%s", profile)
        yield (
            "app_error",
            {
                "message": "Something went wrong while answering. Please try again.",
                "code": "internal_error",
            },
        )
        yield ("done", {"ok": False, "code": "internal_error"})
    finally:
        ctx = tool_server.end_turn(profile)
        if ctx and ctx.remembered:
            logger.info("profile=%s stored %d memory note(s)", profile, len(ctx.remembered))



# ── The second door: one request, one JSON answer ────────────────────────────
#
# Same engine, no event parsing. Streaming earns its complexity when a human is
# watching tokens appear — it fills 8-60s with `status`, then the source rail,
# then text, and a connection that dies at second 40 has still delivered most of
# the answer. It buys nothing when the caller is a service that takes a question
# off a queue and writes the answer somewhere, and it costs that caller the
# whole SSE contract: the bare-string token, sources-replaces-not-appends,
# data:-line reassembly, keepalive frames, reconnect-on-EOF, and errors arriving
# as events AFTER a 200.
#
# So this drains the same generator and answers with a status code. What it
# cannot do is hide the latency: the caller waits the whole turn with no signal,
# and a dropped connection loses the answer entirely rather than partially.

class AskIn(BaseModel):
    """
    The request body for the non-streaming door.

    A body rather than query parameters, deliberately. The streaming endpoint is
    a GET, so everything rides in the URL — which is why a long conversation, or
    a normal-length one in a non-Latin script, can exceed a proxy's URL limit and
    come back as an opaque 400. A POST body has no such ceiling and needs no
    percent-encoding.
    """

    q: str = Field(min_length=1)
    session_id: Optional[str] = None
    max_tokens: int = -1
    pause_personalization: bool = False
    replace_history: bool = False
    client_history: Optional[List[Dict[str, str]]] = None
    doc_ids: Optional[List[str]] = None


class AskOut(BaseModel):
    ok: bool = True
    answer: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    grounding: str = "general_fallback"
    timing_ms: int = 0
    session_id: Optional[str] = None


_FAILURE_STATUS = {
    "agent_unavailable": 503,
    "empty_answer": 502,
    "internal_error": 500,
}


@router.post(
    "/chatbot/api/chats/message",
    tags=["Chats"],
    response_model=AskOut,
    description=(
        "Answer one question and return it whole. The non-streaming door onto the "
        "same turn as GET .../message/stream — same gates, same retrieval, same "
        "citation register, same memory. Prefer this for server-to-server callers; "
        "prefer the stream when a person is watching.\\n\\n"
        "Failures are HTTP statuses, not events: 502 the agent produced no text, "
        "503 the agent is unreachable, plus the shared 401/403/409/429/503 from the "
        "gates. Expect to wait the whole turn — measured at 8-50s, so set a client "
        "timeout of at least 120s, and do not retry a 409.\n\n"
        "Takes a JSON body, not query parameters: the streaming GET carries the "
        "question and history in the URL, which a long or non-Latin-script "
        "conversation can push past a proxy's limit."
    ),
)
async def ask_message(body: AskIn, request: Request):
    q = body.q
    auth_token, user_uuid, profile = await _claim_turn(request, q)
    started = time.monotonic()

    # _turn_events speaks the streaming endpoint's wire types, so the structured
    # body is rendered back into them rather than duplicating the parsing.
    client_history = json.dumps(body.client_history) if body.client_history else None
    doc_ids = ",".join(body.doc_ids) if body.doc_ids else None

    answer_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    grounding = "general_fallback"
    failure: Optional[str] = None

    async for event, data in _turn_events(
        profile=profile,
        auth_token=auth_token,
        user_uuid=user_uuid,
        session_id=body.session_id,
        q=q,
        max_tokens=body.max_tokens,
        pause_personalization=body.pause_personalization,
        replace_history=body.replace_history,
        client_history=client_history,
        doc_ids=doc_ids,
        started=started,
    ):
        if event == "token":
            answer_parts.append(data)
        elif event == "sources":
            sources = data                      # replaces, never appends
        elif event == "grounding":
            grounding = data.get("mode", grounding)
        elif event == "final":
            # The authoritative text, including the non-streaming retry's answer.
            answer_parts = [data.get("text", "")]
        elif event == "app_error":
            failure = data.get("code") or "internal_error"

    if failure:
        raise HTTPException(
            status_code=_FAILURE_STATUS.get(failure, 500),
            detail={"code": failure, "message": "The assistant could not answer this turn."},
        )

    return AskOut(
        answer="".join(answer_parts),
        sources=sources,
        grounding=grounding,
        timing_ms=int((time.monotonic() - started) * 1000),
        session_id=body.session_id,
    )
