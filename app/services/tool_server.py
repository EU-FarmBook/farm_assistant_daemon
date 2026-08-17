# app/services/tool_server.py
"""
The capabilities Hermes is given — two, and no others.

  search_eu_farmbook(query)      retrieve EU-FarmBook material to ground an answer
  remember_about_user(fact)      persist one fact about the user, to MySQL via Django

Hermes holds no credentials for either. It calls an MCP tool, the bridge calls
this service, and this service does the privileged work. No OpenSearch password
and no user JWT ever enters the agent container.

**Why `remember_about_user` exists.** Hermes' built-in memory writes MEMORY.md /
USER.md to the container volume. Personal data may not live there — it belongs in
MySQL under django_euf_admin — so the built-in memory is switched OFF in
config.yaml and this tool replaces it. The agent still decides what is worth
remembering (the mneme model); only the storage moved.

**Turn context.** Both tools need per-request state the agent cannot carry: the
caller's JWT, and somewhere to park retrieved sources for the UI. The bridge tags
each call with the profile it runs as (seeded into its env at provisioning), and
this module keys that state by profile. One user is one profile is one in-flight
turn, so it does not need to be smarter than that.

Consequence before this is ever scaled: turn context is in-process, so this
service is single-replica, and it holds a live token for the duration of a turn.
For a pilot that is the right trade. For anything larger it moves to Valkey.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings
from app.schemas import AskIn, SourceItem
from app.services.context_service import (
    build_context_and_sources,
    estimate_retrieval_quality,
    estimate_semantic_quality,
    filter_items_by_min_score,
)
from app.services.search_service import build_search_payload, collect_os_items

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.tool")

# A turn that has not been touched in this long is over; its context is dropped
# so a stale token can't be used and stale citations can't be attributed to a
# later answer.
_TURN_TTL_SECONDS = 300.0


@dataclass
class TurnContext:
    """
    State for one in-flight turn, including the turn's citation register.

    `sources` accumulates across every retrieval in the turn and the numbering
    the agent sees continues from it. That is not a nicety: an agent that
    searches twice would otherwise be handed two passages both called [1], cite
    one of them, and have the UI render the other — silent mis-citation, which
    for a source-cited assistant is worse than no citation at all.

    `version` increments on every retrieval so the streaming route can re-emit
    the (cumulative) source list as later hops land.
    """
    auth_token: str
    user_uuid: str
    # The user's own words this turn. remember_about_user validates every
    # proposed fact against this: a fact the user did not assert is not a fact.
    user_message: str = ""
    # Note ids in the order they were numbered [M1], [M2]... in the prompt, so
    # the agent can name one to forget.
    note_ids: List[int] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    sources: Optional[List[SourceItem]] = None
    version: int = 0
    remembered: List[str] = field(default_factory=list)

    def register(self, new_sources: List[SourceItem]) -> List[int]:
        """
        Add this hop's sources to the register and return their turn-global
        numbers, 1-based. A source already cited earlier in the turn keeps its
        original number instead of being listed twice.
        """
        if self.sources is None:
            self.sources = []

        numbers: List[int] = []
        for src in new_sources:
            key = src.id or src.url or src.title
            existing = next(
                (
                    i for i, seen in enumerate(self.sources)
                    if (seen.id or seen.url or seen.title) == key
                ),
                None,
            )
            if existing is None:
                self.sources.append(src)
                numbers.append(len(self.sources))
            else:
                numbers.append(existing + 1)

        self.version += 1
        return numbers


_turns: Dict[str, TurnContext] = {}


def begin_turn(
    profile: str, *, auth_token: str, user_uuid: str, user_message: str = ""
) -> None:
    """Open turn context for a profile, discarding anything left from before."""
    _turns[profile] = TurnContext(
        auth_token=auth_token, user_uuid=user_uuid, user_message=user_message,
    )


def set_note_ids(profile: str, note_ids: List[int]) -> None:
    """Record which note id each [M<n>] marker in the prompt refers to."""
    ctx = _live_context(profile)
    if ctx:
        ctx.note_ids = list(note_ids)


def end_turn(profile: str) -> Optional[TurnContext]:
    """Close turn context and hand back what the tools recorded."""
    return _turns.pop(profile, None)


def _live_context(profile: str) -> Optional[TurnContext]:
    ctx = _turns.get(profile)
    if not ctx:
        return None
    if time.monotonic() - ctx.started > _TURN_TTL_SECONDS:
        _turns.pop(profile, None)
        return None
    return ctx


def peek_sources(profile: str) -> Optional[tuple[int, List[SourceItem]]]:
    """
    Non-destructive read of the turn's citation register: (version, sources).

    The streaming route polls this between chunks so citations reach the UI as
    soon as the agent has retrieved, and re-emits when the version moves — which
    is how a second search's sources reach the UI instead of being swallowed.
    """
    ctx = _live_context(profile)
    if not ctx or ctx.sources is None:
        return None
    return ctx.version, list(ctx.sources)


async def search_eu_farmbook(
    query: str,
    profile: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run one EU-FarmBook retrieval and return numbered passages.

    The return shape is the agent's view: numbered passages it is expected to
    cite as [1], [2], ... The matching SourceItems are parked for the UI, because
    the model has no business rendering platform URLs itself.
    """
    ctx = _live_context(profile)
    if not ctx:
        # A tool call outside a turn means the bridge is mis-wired or a turn
        # timed out. Refuse rather than serve an unattributable retrieval.
        return {"ok": False, "error": "No active turn for this profile.", "passages": []}

    k = top_k if isinstance(top_k, int) and top_k > 0 else S.TOP_K
    payload = build_search_payload(AskIn(question=query, top_k=k))

    # Same construction as farm_assistant_um's opensearch_client: Basic auth
    # only when BOTH values are present, and identical headers. When either is
    # blank the request goes out unauthenticated and scout answers 401 — which
    # reads downstream as "the platform has nothing", so log the distinction.
    auth = None
    if S.OPENSEARCH_API_USR and S.OPENSEARCH_API_PWD:
        auth = httpx.BasicAuth(S.OPENSEARCH_API_USR, S.OPENSEARCH_API_PWD)
    else:
        logger.error(
            "OPENSEARCH_API_USR/PWD are not both set — the search request will be "
            "sent unauthenticated and scout will reject it."
        )

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0, verify=S.VERIFY_SSL) as client:
            items = await collect_os_items(client, payload, [1], headers, auth)
    except httpx.HTTPError as e:
        logger.error(
            "Retrieval FAILED for profile=%s (url=%s): %s",
            profile, f"{S.OPENSEARCH_API_URL}{S.OS_RAG_API_PATH}", e,
        )
        # An outage and an empty index are different facts, and the difference
        # reaches the user. Told "no passages", the model says EU-FarmBook has
        # no material on the subject — a false claim about the platform derived
        # from a network error. So say which one this is, explicitly.
        return {
            "ok": False,
            "error": (
                "The EU-FarmBook search service is unreachable right now. This is a "
                "technical fault, NOT evidence that the platform lacks material. Tell "
                "the user search is temporarily unavailable and do not claim anything "
                "about what EU-FarmBook does or does not contain."
            ),
            "passages": [],
        }

    # Per-item junk floor, same threshold v2 applies. This is an OpenSearch score
    # cut, not a judgement about relevance — the judgement is reported below and
    # left to the agent.
    items, filter_stats = filter_items_by_min_score(items, min_score=S.RETRIEVAL_MIN_SCORE)
    if filter_stats["discarded_count"]:
        logger.info(
            "profile=%s score-filtered %s/%s items (threshold %.3f)",
            profile, filter_stats["discarded_count"],
            filter_stats["discarded_count"] + filter_stats["kept_count"],
            filter_stats["min_score_threshold"],
        )

    contexts, sources = build_context_and_sources(
        items=items,
        question=query,
        top_k=k,
        max_context_chars=S.MAX_CONTEXT_CHARS,
    )

    logger.info(
        "Retrieval for profile=%s query=%r -> %d items, %d contexts",
        profile, query[:80], len(items), len(contexts),
    )

    if not contexts:
        ctx.register([])
        return {
            "ok": True,
            "passages": [],
            "quality": {"verdict": "empty"},
            "note": (
                "No EU-FarmBook material matched this query. Try one more search with "
                "different or broader terms if the question warrants it; otherwise say "
                "plainly that EU-FarmBook has no material on this. Do not substitute "
                "your own knowledge for platform sources."
            ),
        }

    quality = _assess(query, items)
    numbers = ctx.register(sources)

    return {
        "ok": True,
        "passages": [
            {
                "n": numbers[i] if i < len(numbers) else i + 1,
                "text": ctx_text,
                "title": (sources[i].title if i < len(sources) else None),
            }
            for i, ctx_text in enumerate(contexts)
        ],
        "quality": quality,
        "note": (
            "Cite these passages by the `n` shown. Those numbers are stable for the whole "
            "conversation turn — if you search again, earlier numbers keep their meaning."
            + (
                " These results look weak for the question asked. Consider one more search "
                "with more specific terms before answering, and if it stays weak, say that "
                "EU-FarmBook has little on this rather than over-claiming."
                if quality["verdict"] == "weak" else ""
            )
        ),
    }


def _assess(query: str, items: list) -> Dict[str, Any]:
    """
    Score how well this hop answered the query, and say so in the tool result.

    v2 uses these same numbers to silently DROP weak contexts. v3 deliberately
    does not drop: the whole premise of the agent route is that the model decides
    what to do about weak retrieval, and it cannot decide well with no signal.
    So the verdict is reported and the decision is left upstream.
    """
    semantic = (
        estimate_semantic_quality(items, top_n=3)
        if (S.RELEVANCE_MODE or "").strip().lower() == "semantic"
        else None
    )

    if semantic is not None:
        score, mode, threshold = semantic, "semantic", S.SEMANTIC_DROP_THRESHOLD
    else:
        score, mode, threshold = (
            estimate_retrieval_quality(query, items, top_n=3),
            "overlap",
            S.RETRIEVAL_DROP_THRESHOLD,
        )

    return {
        "score": round(float(score), 3),
        "mode": mode,
        "threshold": threshold,
        "verdict": "strong" if score >= threshold else "weak",
    }


async def remember_about_user(fact: str, profile: str) -> Dict[str, Any]:
    """
    Persist one durable fact about the user to MySQL, through Django.

    Written with the CALLER's token, so the row lands on the right account and
    Django's own ownership checks still apply — the adapter never gets a
    privileged write path into other users' memory.
    """
    from app.services import memory_guard, memory_service

    ctx = _live_context(profile)
    if not ctx:
        return {"ok": False, "error": "No active turn for this profile."}

    text = (fact or "").strip()
    if not text:
        return {"ok": False, "error": "Empty fact."}

    # The gate: the user's own message must assert this about them. Without it,
    # the topic of a question becomes a durable fact about the person asking.
    allowed, reason = await memory_guard.is_supported_by_user(text, ctx.user_message)
    if not allowed:
        logger.info("Refused memory %r for profile=%s: %s", text[:80], profile, reason)
        return {
            "ok": False,
            "error": (
                f"Not stored — {reason}. Only store what the user stated about "
                "themselves in their own message."
            ),
        }

    ok = await memory_service.add_note(ctx.auth_token, text)
    if ok:
        ctx.remembered.append(text)
    return {"ok": ok}


async def forget_about_user(marker: str, profile: str) -> Dict[str, Any]:
    """
    Delete one remembered note, named by its [M<n>] marker from the prompt.

    Without this the agent can only ADD. A user correcting "I'm in the
    Netherlands, not Italy" would end up with both notes stored and the
    contradiction surfacing in every later answer — which is precisely what
    happened in the pilot.
    """
    from app.services import memory_service

    ctx = _live_context(profile)
    if not ctx:
        return {"ok": False, "error": "No active turn for this profile."}

    token = (marker or "").strip().upper().lstrip("[").rstrip("]").lstrip("M")
    if not token.isdigit():
        return {"ok": False, "error": "Name the note by its marker, e.g. M2."}

    index = int(token)
    if not 1 <= index <= len(ctx.note_ids):
        return {"ok": False, "error": f"No remembered note numbered M{index} this turn."}

    note_id = ctx.note_ids[index - 1]
    ok = await memory_service.delete_note(ctx.auth_token, note_id)
    if ok:
        logger.info("Forgot note id=%s (M%s) for profile=%s", note_id, index, profile)
    return {"ok": ok}
