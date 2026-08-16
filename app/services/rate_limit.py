# app/services/rate_limit.py
"""
Per-user turn limits.

Necessary once access is open. A chat turn here is not one model call: it is an
agent loop of up to HERMES_MAX_ITERATIONS calls, each carrying the system prompt,
the user's remembered profile and any retrieved passages, all billed to one
provider key. Without a limit, a single authenticated account — or a script
holding a valid token — can spend without bound, and the first sign of it is the
invoice.

Counted per VERIFIED uuid, never per IP: the uuid is what the adapter trusts for
everything else, and an IP is both shared (institutional NAT) and trivially
changed.

In-process, like the turn context in tool_server, because this service is
single-replica by design. If it is ever scaled out, both move to Valkey
together — a per-replica limit on N replicas is an N-times-larger limit, so
that migration is not optional.
"""

import logging
import time
from collections import deque
from typing import Deque, Dict, Tuple

from app.config import get_settings

logger = logging.getLogger("farm-assistant-hermes.ratelimit")

_MINUTE = 60.0
_DAY = 86400.0

# uuid -> timestamps of recent turns, newest last.
_turns: Dict[str, Deque[float]] = {}

# Don't let the map grow forever on a long-lived process.
_MAX_TRACKED_USERS = 50_000


class RateLimited(Exception):
    """The caller has exceeded their allowance."""

    def __init__(self, retry_after_seconds: int, scope: str):
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.scope = scope
        super().__init__(f"Rate limit exceeded ({scope})")


def _prune(stamps: Deque[float], now: float) -> None:
    while stamps and now - stamps[0] > _DAY:
        stamps.popleft()


def check_and_record(user_uuid: str) -> None:
    """
    Record a turn for `user_uuid`, or raise RateLimited.

    Called once per turn, before any model call — the point is to refuse before
    spending, not after.
    """
    settings = get_settings()
    if not settings.RATE_LIMIT_ENABLED or not user_uuid:
        return

    now = time.monotonic()
    stamps = _turns.get(user_uuid)
    if stamps is None:
        if len(_turns) >= _MAX_TRACKED_USERS:
            # Drop the coldest entry rather than grow without bound. Worst case
            # a returning user gets a fresh allowance, which is the safe way to
            # be wrong here.
            oldest = min(_turns, key=lambda k: _turns[k][-1] if _turns[k] else 0.0)
            _turns.pop(oldest, None)
        stamps = _turns[user_uuid] = deque()

    _prune(stamps, now)

    per_day = settings.RATE_LIMIT_TURNS_PER_DAY
    if per_day > 0 and len(stamps) >= per_day:
        retry = _DAY - (now - stamps[0])
        logger.info("uuid=%s hit the daily limit (%s turns)", user_uuid, per_day)
        raise RateLimited(retry, "day")

    per_min = settings.RATE_LIMIT_TURNS_PER_MIN
    if per_min > 0:
        recent = sum(1 for t in stamps if now - t <= _MINUTE)
        if recent >= per_min:
            oldest_in_window = next(t for t in stamps if now - t <= _MINUTE)
            retry = _MINUTE - (now - oldest_in_window)
            logger.info("uuid=%s hit the per-minute limit (%s turns)", user_uuid, per_min)
            raise RateLimited(retry, "minute")

    stamps.append(now)


def usage(user_uuid: str) -> Tuple[int, int]:
    """(turns in the last minute, turns in the last day) — for logging/debug."""
    stamps = _turns.get(user_uuid)
    if not stamps:
        return 0, 0
    now = time.monotonic()
    return sum(1 for t in stamps if now - t <= _MINUTE), len(stamps)


def reset() -> None:
    """Test helper."""
    _turns.clear()
