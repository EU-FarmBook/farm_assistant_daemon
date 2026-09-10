# app/services/auth_service.py
#
# Verified identity resolution for Bearer tokens.
#
# Tokens are minted by django_euf_admin (SimpleJWT HS256 + a token_version
# revocation counter). FA does not hold the signing key, so identity is
# resolved by INTROSPECTION: the raw token is POSTed to Django's
# /fastapi/validate_access_token/, which checks signature, expiry, AND
# token_version (i.e. server-side revocation — something a local signature
# check could never see). Verified verdicts are cached in-process so the
# steady-state cost is a dict lookup.
#
# Behavioral contract — deliberately compatible with the old unverified
# decode so existing flows keep working:
#   - No/blank Authorization header  -> None (anonymous, unchanged).
#   - Structurally invalid token     -> None without calling Django (unchanged).
#   - Valid-looking but FORGED/expired/revoked token -> None (NEW: these used
#     to be trusted; anonymous/401 now, via the endpoints' existing paths).
#   - Introspection backend unreachable -> cached verdict when available,
#     otherwise fall back to the unverified decode WITH a warning. Django
#     being down already degrades history/persistence; hard-401ing chat and
#     files during a blip would break more than it protects.
#   - No auth backend configured (bare local dev) or AUTH_TOKEN_INTROSPECTION
#     disabled -> unverified decode, as before.

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import Optional

import httpx

from app.config import get_settings

S = get_settings()
logger = logging.getLogger("farm-assistant.auth")

VALID_VERDICT_TTL_SECONDS = 300.0
INVALID_VERDICT_TTL_SECONDS = 60.0
_CACHE_MAX_ENTRIES = 4096
_INTROSPECT_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)

# token sha256 -> (user_uuid | None, monotonic expiry)
_verdicts: dict[str, tuple[Optional[str], float]] = {}
_lock = asyncio.Lock()


def decode_token_uuid(auth_header: Optional[str]) -> Optional[str]:
    """
    Extract the uuid/user_id/sub claim from a Bearer JWT WITHOUT verifying it.
    Claim extraction only — never treat the result as authenticated identity
    unless it came through resolve_user_uuid().
    
DIVERGENCE FROM farm_assistant_um: this file is otherwise a verbatim copy, but
the two introspection-failure paths below now DENY instead of falling back to
the token's own claim (see _may_trust_unverified). The original behaviour is an
impersonation window on any public host whenever Django is unhealthy. If the
sibling service still has the original, it needs the same fix — that is a real
finding about v2, not a divergence to reconcile away.
"""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        token_data = json.loads(base64.urlsafe_b64decode(payload))
        user_id = token_data.get("uuid") or token_data.get("user_id") or token_data.get("sub")
        return str(user_id) if user_id else None
    except Exception:
        return None


def decode_token_claims(auth_header: Optional[str]) -> dict:
    """
    Return the JWT payload WITHOUT verifying it.

    Only safe to act on for a token that resolve_user_uuid() has already
    verified: introspection checks the signature, so the claims of a token that
    passed are the issuer's, not the caller's. Never call this on its own to
    make an access decision.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        return {}
    token = auth_header[7:]
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        data = json.loads(base64.urlsafe_b64decode(payload))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def decode_token_email(auth_header: Optional[str]) -> Optional[str]:
    """
    Best-effort email from a JWT. Claim naming varies by issuer, and these
    tokens are minted by the public API rather than by this codebase, so try the
    usual spellings and return None rather than guessing.
    """
    claims = decode_token_claims(auth_header)
    for key in ("email", "user_email", "mail", "preferred_username", "username"):
        value = claims.get(key)
        if isinstance(value, str) and "@" in value:
            return value.strip().lower()
    return None


def decode_token_first_name(auth_header: Optional[str]) -> Optional[str]:
    """
    Best-effort FIRST name from a verified JWT. Never the surname, never the email.

    Only safe on a token resolve_user_uuid() has already verified. Claim naming
    varies by issuer and these tokens are minted by the public API rather than
    here, so several spellings are tried and None is returned rather than a guess.

    Deliberately just the first name: "Hello Anna" is worth something to a user,
    while a full name plus an email address turns a pseudonymous profile at the
    inference provider into an identified person, for no assistant value at all.
    """
    claims = decode_token_claims(auth_header)
    for key in ("first_name", "given_name", "name", "full_name"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip() and "@" not in value:
            first = value.strip().split()[0]
            # Guard against a uuid or slug landing in a name claim.
            if 1 < len(first) <= 40 and any(c.isalpha() for c in first):
                return first
    return None


def _auth_base_url() -> str:
    return (S.AUTH_BACKEND_URL or S.CHAT_BACKEND_URL or "").rstrip("/")


def _cache_get(key: str) -> Optional[tuple[Optional[str], float]]:
    entry = _verdicts.get(key)
    if entry and entry[1] > time.monotonic():
        return entry
    return None


def _cache_put(key: str, user_uuid: Optional[str], ttl: float) -> None:
    if len(_verdicts) >= _CACHE_MAX_ENTRIES:
        now = time.monotonic()
        for stale in [k for k, (_, exp) in _verdicts.items() if exp <= now]:
            _verdicts.pop(stale, None)
        while len(_verdicts) >= _CACHE_MAX_ENTRIES:
            _verdicts.pop(next(iter(_verdicts)), None)
    _verdicts[key] = (user_uuid, time.monotonic() + ttl)


def _may_trust_unverified() -> bool:
    """
    May an introspection FAILURE fall back to the token's own uuid claim?

    Only in bare local dev. Everywhere else a failure must DENY.

    The original code chose availability: any transport error or unexpected
    status returned the claimed uuid so the assistant kept answering through a
    Django blip. But this uuid is the only thing deciding which agent, whose
    memory and whose transcript a request reaches — so on a public host that
    trade means anyone can impersonate any user for as long as Django is
    unhealthy, by sending an unsigned `alg=none` token carrying their uuid.
    An attacker can wait for that window, or provoke it.

    The FA_ENV gate in main.py does not cover this: auth_is_verified() checks
    that introspection is CONFIGURED, never that it is reachable.
    """
    return (S.FA_ENV or "local").strip().lower() == "local"


async def resolve_user_uuid(auth_header: Optional[str]) -> Optional[str]:
    """
    Resolve the authenticated user's uuid from an Authorization header.
    Returns None for anonymous/invalid tokens — callers keep their existing
    "no uuid" handling (anonymous chat, 401 on owner-scoped endpoints).
    """
    claimed_uuid = decode_token_uuid(auth_header)
    if not claimed_uuid:
        return None

    base_url = _auth_base_url()
    if not S.AUTH_TOKEN_INTROSPECTION or not base_url:
        # Bare local dev or explicit kill switch: old behavior.
        return claimed_uuid

    token = auth_header[7:]  # decode_token_uuid proved the "Bearer " prefix
    key = hashlib.sha256(token.encode("utf-8")).hexdigest()

    cached = _cache_get(key)
    if cached:
        return cached[0]

    async with _lock:
        cached = _cache_get(key)
        if cached:
            return cached[0]
        try:
            async with httpx.AsyncClient(timeout=_INTROSPECT_TIMEOUT, verify=S.VERIFY_SSL) as client:
                r = await client.post(
                    f"{base_url}/fastapi/validate_access_token/",
                    json={"access_token": token},
                )
        except httpx.HTTPError as e:
            if _may_trust_unverified():
                logger.warning(
                    "Token introspection unreachable (%s); falling back to unverified "
                    "decode because FA_ENV=local.", e
                )
                return claimed_uuid
            # Deny. Not cached: an outage must not be remembered as a verdict
            # about this token, or recovery would lag the outage by the TTL.
            logger.error(
                "Token introspection unreachable (%s); REFUSING the request rather than "
                "trusting an unverified token.", e
            )
            return None

        if r.status_code == 200:
            _cache_put(key, claimed_uuid, VALID_VERDICT_TTL_SECONDS)
            return claimed_uuid
        if r.status_code in (400, 401, 403):
            logger.info("Token introspection rejected a token (HTTP %s).", r.status_code)
            _cache_put(key, None, INVALID_VERDICT_TTL_SECONDS)
            return None

        # Unexpected upstream state (5xx, proxy errors). Same rule as a
        # transport failure: identity is not something to guess at.
        if _may_trust_unverified():
            logger.warning(
                "Token introspection returned HTTP %s; falling back to unverified decode "
                "because FA_ENV=local.", r.status_code
            )
            return claimed_uuid
        logger.error(
            "Token introspection returned HTTP %s; REFUSING the request rather than "
            "trusting an unverified token.", r.status_code
        )
        return None
