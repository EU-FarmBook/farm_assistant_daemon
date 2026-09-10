# app/routers/_access.py
"""
The gate for endpoints that SPEND on the inference provider.

Three endpoints — /chatbot/api/follow-ups, /chatbot/api/export-intent and
/chatbot/api/chats/<id>/title — each make a paid completion on LLM_API_KEY, and
each required nothing more than a resolvable uuid. Neither the pilot gate nor
the rate limiter was applied, so with a closed roster a user who is deliberately
NOT on it (and who gets a hard 403 from the streaming endpoint) could still POST
here in a loop and spend, while the README named RATE_LIMIT_* as the only bound
on spend.

Two deliberate choices:

* It does NOT raise. All three endpoints document that they degrade rather than
  fail — empty chips, `format: null`, the chat keeping its default name — because
  each is a convenience sitting beside an answer that was already delivered.
  Turning one into a 403 would make the UI worse without making anything safer.
  What matters is not spending; the caller reports the reason instead.
* It meters the AUX bucket, not the turn bucket. Each of these is one cheap
  completion TRIGGERED BY a turn, so charging them to the agent-turn allowance
  would have silently halved the documented 6 turns/min.
"""

from typing import Optional, Tuple

from fastapi import Request

from app.services import rate_limit
from app.services.auth_service import decode_token_email, resolve_user_uuid
from app.services.profile_registry import ProfileNotProvisioned, resolve_profile


async def spend_allowed(request: Request) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (auth_token, user_uuid, refusal).

    `refusal` is None when the call may proceed, otherwise a short stable reason:
    "unauthenticated", "not_in_pilot" or "rate_limited".
    """
    auth_token = request.headers.get("Authorization", "")
    user_uuid = await resolve_user_uuid(auth_token) if auth_token else None
    if not user_uuid:
        return auth_token, None, "unauthenticated"

    try:
        # Same email-claim rule as the streaming endpoint: read it only after
        # resolve_user_uuid has verified the token.
        resolve_profile(user_uuid, email=decode_token_email(auth_token))
    except ProfileNotProvisioned:
        return auth_token, user_uuid, "not_in_pilot"

    try:
        rate_limit.check_and_record(user_uuid, bucket=rate_limit.AUX_BUCKET)
    except rate_limit.RateLimited:
        return auth_token, user_uuid, "rate_limited"

    return auth_token, user_uuid, None
