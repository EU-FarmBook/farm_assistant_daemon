"""
Tests for the two properties that make this service safe to point at real users:
a caller cannot reach another user's agent, and a caller cannot choose their own
memory scope.

Everything else in the pilot is an experiment. These are not.
"""

import pytest

from app.config import Settings
from app.security import hash_key, path_requires_key, resolve_api_key_label
from app.services import hermes_client, profile_registry
from app.services.profile_registry import ProfileNotProvisioned


# --- The allowlist -------------------------------------------------------
#
# These cover the roster decision only; provisioning has its own file. Hence
# `provision=False` — creating directories is not what is under test here.

def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


@pytest.fixture
def rostered(monkeypatch):
    s = _settings(HERMES_PILOT_UUIDS="uuid-a,uuid-b")
    monkeypatch.setattr(profile_registry, "get_settings", lambda: s)
    return s


def test_rostered_uuid_resolves_to_its_own_profile(rostered):
    assert profile_registry.resolve_profile("uuid-a", provision=False) == "uuid-a"
    assert profile_registry.resolve_profile("uuid-b", provision=False) == "uuid-b"


def test_uuid_off_the_roster_is_refused_not_defaulted(rostered):
    # The failure mode this guards against is a "default profile" fallback:
    # everyone outside the pilot would land in ONE agent and share its memory.
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile("uuid-stranger", provision=False)


def test_anonymous_is_refused(rostered):
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile(None, provision=False)
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile("", provision=False)


def test_traversal_shaped_profile_name_is_rejected(monkeypatch):
    s = _settings(HERMES_PROFILE_MAP="uuid-a:../../etc")
    monkeypatch.setattr(profile_registry, "get_settings", lambda: s)
    # A bad name is config error, not user input — but it would become a URL path
    # segment AND a directory name, so it must never reach either.
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile("uuid-a", provision=False)


@pytest.mark.parametrize("name", ["alice", "a", "pilot_user-1", "u" * 64,
                                  "45b75f62-3fa3-4b18-8593-1411f110a98e"])
def test_valid_profile_names(name):
    assert profile_registry.is_valid_profile_name(name)


@pytest.mark.parametrize("name", ["", "Alice", "-alice", "a/b", "a..b", "u" * 65, "alice ", "p:1"])
def test_invalid_profile_names(name):
    assert not profile_registry.is_valid_profile_name(name)


# --- The memory scope header --------------------------------------------

def test_session_key_is_the_verified_uuid():
    headers = hermes_client.build_headers(user_uuid="uuid-a", session_id="sess-1")
    assert headers["X-Hermes-Session-Key"] == "uuid-a"
    assert headers["X-Hermes-Session-Id"] == "sess-1"


def test_build_headers_has_no_client_supplied_scope_path():
    # Hermes will honour ANY session key from a caller holding the API key, so
    # the adapter must have no parameter that lets a client name its own scope.
    # If this signature ever grows one, that is the bug.
    import inspect

    params = set(inspect.signature(hermes_client.build_headers).parameters)
    assert params == {"user_uuid", "session_id"}


def test_profile_path_is_per_profile(monkeypatch):
    monkeypatch.setattr(hermes_client, "S", Settings(HERMES_MULTIPLEX_PROFILES=True, _env_file=None))
    assert hermes_client._base_path("alice") == "/p/alice/v1"
    assert hermes_client._base_path("bob") == "/p/bob/v1"


# --- The caller-key gate -------------------------------------------------

def test_api_key_lookup_matches_only_the_right_key():
    key = "eufb_test_abc"
    keys = {hash_key(key): "frontend-v3"}
    assert resolve_api_key_label(key, keys) == "frontend-v3"
    assert resolve_api_key_label("eufb_test_abd", keys) is None
    assert resolve_api_key_label("", keys) is None


def test_chat_paths_require_a_key():
    assert path_requires_key("/chatbot/api/chats/message/stream", "GET")
    assert path_requires_key("/chatbot/api/users/me/memory/documents", "GET")
    # Preflight and health stay open, or the browser and the orchestrator break.
    assert not path_requires_key("/chatbot/api/chats/message/stream", "OPTIONS")
    assert not path_requires_key("/health", "GET")


# --- Fail-closed auth ----------------------------------------------------

def test_blank_backend_resolves_from_fa_env_not_to_nothing():
    # A blank backend URL makes auth_service trust an UNVERIFIED JWT decode, so
    # it must never stay blank. farm_assistant_um resolves it from FA_ENV; this
    # service must do the same or a deployment silently accepts forged tokens.
    assert Settings(FA_ENV="prd", _env_file=None).CHAT_BACKEND_URL == (
        "https://backend-admin.prd.farmbook.ugent.be"
    )
    assert Settings(FA_ENV="dev", _env_file=None).CHAT_BACKEND_URL == (
        "https://backend-admin.dev.farmbook.ugent.be"
    )


def test_auth_is_verified_reports_the_kill_switch():
    assert Settings(FA_ENV="prd", _env_file=None).auth_is_verified()
    assert not Settings(FA_ENV="prd", AUTH_TOKEN_INTROSPECTION=False, _env_file=None).auth_is_verified()
