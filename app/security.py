# app/security.py
"""
API-key gate: a second credential (a caller key from a managed pool) required on
the programmatic chat API, on top of the login JWT. Keys are held HASHED in
config (CHAT_API_KEYS = csv of `label:sha256hex`); the plaintext key lives only
with the holder. See API_KEY_GATING_DESIGN.md.

This module is pure/stateless so it is unit-testable without an app or network.
main.py installs `api_key_middleware` which calls into here.
"""

import hashlib
import hmac

# API-path prefixes that require a key. Everything else (UI pages, /health, /docs,
# /static, /openapi.json) is open. Auth endpoints are carved out below — you must
# be able to log in / mint a token WITHOUT already holding a key.
_PROTECTED_PREFIXES = ("/chatbot/api/", "/ask", "/proxy/", "/files/")

# Open even though they match a protected prefix: the login + token endpoints.
_OPEN_PATHS = frozenset({
    "/chatbot/api/auth/login",
    "/chatbot/api/auth/token",
})


def hash_key(plaintext: str) -> str:
    """SHA-256 hex digest of a plaintext API key (lowercased hex)."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def path_requires_key(path: str, method: str) -> bool:
    """
    True if a request to (path, method) must carry a valid API key.

    CORS preflight (OPTIONS) is always exempt — browsers cannot attach custom
    headers to a preflight, and the actual request that follows is still gated.
    """
    if method.upper() == "OPTIONS":
        return False
    if path in _OPEN_PATHS:
        return False
    return path.startswith(_PROTECTED_PREFIXES)


def resolve_api_key_label(presented: str, keys_map: dict[str, str]) -> str | None:
    """
    Return the caller label for a presented plaintext key, or None if it is
    missing/unknown. Comparison is constant-time per candidate (hmac.compare_digest
    over the hex digests) to avoid a timing oracle; the pool is tiny so the linear
    scan is negligible.
    """
    if not presented or not keys_map:
        return None
    digest = hash_key(presented)
    for known_digest, label in keys_map.items():
        if hmac.compare_digest(digest, known_digest):
            return label
    return None
