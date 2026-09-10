"""
The two doors an external application uses, and the browser gate.

Both doors must run ONE turn implementation: the gates, the citation register,
the memory block and the rate limit have to happen once and identically, or the
non-streaming door becomes a second place for bugs to live.
"""

import inspect

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.routers import ask


def test_both_doors_consume_the_same_generator():
    stream = inspect.getsource(ask.stream_message)
    post = inspect.getsource(ask.ask_message)
    assert "_turn_events(" in stream
    assert "_turn_events(" in post
    # Neither may re-implement the turn.
    for src in (stream, post):
        assert "stream_chat(" not in src
        assert "search_eu_farmbook" not in src


def test_both_doors_apply_the_same_gates():
    for fn in (ask.stream_message, ask.ask_message):
        assert "_claim_turn(request, q)" in inspect.getsource(fn)
    gates = inspect.getsource(ask._claim_turn)
    for expected in ("401", "403", "429", "409", "503"):
        assert expected in gates, f"gate {expected} missing"


def test_the_post_door_maps_failures_to_status_codes():
    """The point of the second door: errors are statuses, not events after a 200."""
    assert ask._FAILURE_STATUS == {
        "agent_unavailable": 503,
        "empty_answer": 502,
        "internal_error": 500,
    }
    src = inspect.getsource(ask.ask_message)
    assert "raise HTTPException(" in src
    assert 'sources = data' in src            # replaces, never appends


def test_the_post_door_prefers_final_over_streamed_tokens():
    """
    `final` carries the answer from the non-streaming retry too, so it must win
    over accumulated tokens rather than being appended to them.
    """
    src = inspect.getsource(ask.ask_message)
    assert 'answer_parts = [data.get("text", "")]' in src


# --- CORS ----------------------------------------------------------------

def _client(monkeypatch, origins: str):
    import app.main as main

    s = Settings(CORS_ALLOW_ORIGINS=origins, REQUIRE_API_KEY=False, _env_file=None)
    monkeypatch.setattr(main, "S", s)
    # The middleware stack is built at import, so rebuild the app for this test.
    import importlib

    reloaded = importlib.reload(main)
    monkeypatch.setattr(reloaded, "S", s)
    return reloaded


def test_no_cors_by_default():
    """The default stays a server-to-server API: no middleware, no headers."""
    assert Settings(_env_file=None).cors_origins() == []


def test_a_wildcard_is_refused_not_honoured():
    """This service is handed platform JWTs; "any origin" is not an accident to allow."""
    assert Settings(CORS_ALLOW_ORIGINS="*", _env_file=None).cors_origins() == []
    assert Settings(CORS_ALLOW_ORIGINS="https://a.example,*", _env_file=None).cors_origins() == [
        "https://a.example"
    ]


def test_origins_are_normalised():
    parsed = Settings(
        CORS_ALLOW_ORIGINS=" https://a.example/ , https://b.example ", _env_file=None
    ).cors_origins()
    assert parsed == ["https://a.example", "https://b.example"]


def test_a_preflight_is_answered_when_an_allowlist_is_set(monkeypatch):
    """
    A browser cannot attach X-API-Key to an OPTIONS request. The key gate always
    exempted OPTIONS, but exempting is not answering — nothing answered it, so a
    browser client was impossible. CORS must therefore sit OUTSIDE the gate.
    """
    import importlib
    import os

    os.environ["CORS_ALLOW_ORIGINS"] = "https://app.example"
    os.environ["REQUIRE_API_KEY"] = "false"
    try:
        import app.config
        import app.main

        app.config.get_settings.cache_clear()
        main = importlib.reload(app.main)
        with TestClient(main.app) as client:
            r = client.options(
                "/chatbot/api/chats/message/stream",
                headers={
                    "Origin": "https://app.example",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "authorization,x-api-key",
                },
            )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "https://app.example"
        allowed = r.headers["access-control-allow-headers"].lower()
        assert "authorization" in allowed and "x-api-key" in allowed
    finally:
        os.environ.pop("CORS_ALLOW_ORIGINS", None)
        os.environ.pop("REQUIRE_API_KEY", None)
        import app.config
        import app.main

        app.config.get_settings.cache_clear()
        importlib.reload(app.main)


def test_an_unlisted_origin_gets_no_allow_header(monkeypatch):
    import importlib
    import os

    os.environ["CORS_ALLOW_ORIGINS"] = "https://app.example"
    try:
        import app.config
        import app.main

        app.config.get_settings.cache_clear()
        main = importlib.reload(app.main)
        with TestClient(main.app) as client:
            r = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    finally:
        os.environ.pop("CORS_ALLOW_ORIGINS", None)
        import app.config
        import app.main

        app.config.get_settings.cache_clear()
        importlib.reload(app.main)
