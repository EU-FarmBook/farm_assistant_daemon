# app/services/hermes_client.py
"""
The one place that talks to the Hermes agent.

Two things here are load-bearing:

1. **Profile routing.** With `gateway.multiplex_profiles` on, a request to
   `/p/<profile>/v1/chat/completions` runs as that profile, with that profile's
   own MEMORY.md and USER.md. The profile comes from profile_registry, which
   derives it from the introspected JWT.

2. **X-Hermes-Session-Key.** Hermes accepts a caller-supplied long-term memory
   scope, and only requires that the caller hold the API key. That makes it a
   header a malicious client would love to set. It is built HERE, from the
   verified uuid, and any inbound value is discarded — see build_headers().
"""

import json
import logging
from typing import AsyncIterator, Dict, List, Optional

import httpx

from app.config import get_settings

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.client")


class HermesUnavailable(Exception):
    """Hermes could not be reached or returned a non-2xx before streaming."""


def _base_path(profile: str) -> str:
    """
    Profile-prefixed base path. When multiplexing is off we fall back to the
    bare /v1 path — correct only for a single-user local run, which is why
    HERMES_MULTIPLEX_PROFILES defaults to true and the README says not to
    turn it off with more than one pilot user.
    """
    if not S.HERMES_MULTIPLEX_PROFILES:
        return "/v1"
    return f"/p/{profile}/v1"


def build_headers(
    *,
    user_uuid: str,
    session_id: Optional[str],
) -> Dict[str, str]:
    """
    Headers for a Hermes call.

    `X-Hermes-Session-Key` scopes long-term memory and is derived from the
    VERIFIED uuid. `X-Hermes-Session-Id` scopes the short-term transcript and
    tracks the platform's chat session, so a new chat starts a new transcript
    while the memory scope stays put — which is exactly the split Hermes
    documents between the two headers.

    Note there is no parameter for "session key the client asked for". That is
    intentional and should stay that way.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {S.HERMES_API_KEY}",
        "X-Hermes-Session-Key": user_uuid,
    }
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id
    return headers


async def stream_chat(
    *,
    profile: str,
    user_uuid: str,
    session_id: Optional[str],
    messages: List[Dict[str, str]],
    model: str = "hermes-agent",
) -> AsyncIterator[str]:
    """
    Stream assistant text from Hermes as plain content deltas.

    Yields only the text; the caller re-frames it into the platform's SSE
    contract (`token` / `final` / `done`). Tool calls are executed inside Hermes
    and never surface here — from the adapter's point of view an agent turn is
    one long completion that happens to take longer than a RAG turn.
    """
    url = f"{S.HERMES_API_URL}{_base_path(profile)}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    timeout = httpx.Timeout(
        connect=10.0,
        # An agent loop with a retrieval tool is legitimately slow: several
        # model calls plus an OpenSearch round trip. Read timeout here is
        # per-chunk, not per-request, so this bounds silence, not duration.
        read=S.HERMES_REQUEST_TIMEOUT_SECONDS,
        write=30.0,
        pool=10.0,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=build_headers(user_uuid=user_uuid, session_id=session_id),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:500]
                    logger.error("Hermes returned HTTP %s: %s", response.status_code, body)
                    raise HermesUnavailable(f"Hermes returned HTTP {response.status_code}")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield delta
    except httpx.HTTPError as e:
        logger.error("Hermes stream failed: %s", e)
        raise HermesUnavailable(str(e)) from e


async def complete_chat(
    *,
    profile: str,
    user_uuid: str,
    session_id: Optional[str],
    messages: List[Dict[str, str]],
    model: str = "hermes-agent",
) -> str:
    """
    One non-streaming completion, returning the whole answer.

    Used as a fallback when the streaming endpoint yields nothing. Hermes'
    streaming path has been observed emitting a role delta, an empty delta and
    `finish_reason: stop` while the identical non-streaming call returns real
    content — including error text the stream silently dropped. Rather than
    depend on that, take the answer in one piece and hand it to the client as a
    single chunk: worse typing animation, an answer instead of an empty bubble.
    """
    url = f"{S.HERMES_API_URL}{_base_path(profile)}/chat/completions"
    timeout = httpx.Timeout(
        connect=10.0, read=S.HERMES_REQUEST_TIMEOUT_SECONDS, write=30.0, pool=10.0
    )

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
            response = await client.post(
                url,
                json={"model": model, "messages": messages, "stream": False},
                headers=build_headers(user_uuid=user_uuid, session_id=session_id),
            )
    except httpx.HTTPError as e:
        logger.error("Hermes non-streaming call failed: %s", e)
        raise HermesUnavailable(str(e)) from e

    if response.status_code >= 400:
        logger.error("Hermes returned HTTP %s: %s", response.status_code, response.text[:500])
        raise HermesUnavailable(f"Hermes returned HTTP {response.status_code}")

    try:
        choices = (response.json() or {}).get("choices") or []
        return ((choices[0].get("message") or {}).get("content") or "") if choices else ""
    except (ValueError, IndexError, AttributeError):
        logger.error("Could not parse Hermes response: %s", response.text[:300])
        return ""


async def health() -> bool:
    """Liveness of the agent, for this service's own /health."""
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=S.VERIFY_SSL) as client:
            r = await client.get(f"{S.HERMES_API_URL}/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False
