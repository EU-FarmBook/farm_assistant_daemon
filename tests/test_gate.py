"""
Tests for the pilot gate.

The gate has three inputs and one rule that outranks all of them: with nothing
configured, nobody gets in. An access gate that fails OPEN when misconfigured is
worse than no gate, because it looks like it is working.
"""

import pytest

from app.config import Settings
from app.services import profile_registry
from app.services.auth_service import decode_token_email

UUID_A = "45b75f62-3fa3-4b18-8593-1411f110a98e"


def _use(monkeypatch, **kwargs):
    s = Settings(_env_file=None, **kwargs)
    monkeypatch.setattr(profile_registry, "get_settings", lambda: s)
    # The roster file is cached; drop it so each test reads its own fixture.
    monkeypatch.setattr(profile_registry, "_file_roster_cache", None)
    monkeypatch.setattr(profile_registry, "_file_roster_read_at", 0.0)
    return s


def test_nothing_configured_admits_nobody(monkeypatch):
    _use(monkeypatch)
    assert not profile_registry.is_pilot_user(UUID_A, "someone@ugent.be")


def test_domain_admits_without_knowing_the_uuid(monkeypatch):
    _use(monkeypatch, HERMES_PILOT_EMAIL_DOMAINS="ugent.be,nexavion.com")
    assert profile_registry.is_pilot_user(UUID_A, "pranav@ugent.be")
    assert profile_registry.is_pilot_user(UUID_A, "someone@nexavion.com")
    assert not profile_registry.is_pilot_user(UUID_A, "outsider@gmail.com")
    # No email claim on the token -> the domain rule cannot apply.
    assert not profile_registry.is_pilot_user(UUID_A, None)


def test_domain_matching_is_exact_not_suffix(monkeypatch):
    _use(monkeypatch, HERMES_PILOT_EMAIL_DOMAINS="ugent.be")
    # "notugent.be" ends with "ugent.be"; a naive endswith() would admit it.
    assert not profile_registry.is_pilot_user(UUID_A, "attacker@notugent.be")
    assert not profile_registry.is_pilot_user(UUID_A, "attacker@ugent.be.evil.com")


def test_roster_file_is_read_and_reread(monkeypatch, tmp_path):
    roster = tmp_path / "pilot-roster.txt"
    roster.write_text("# pilot\n\n" + UUID_A + "   # pranav\n", encoding="utf-8")
    _use(monkeypatch, HERMES_PILOT_ROSTER_FILE=str(roster))

    assert profile_registry.is_pilot_user(UUID_A)
    assert not profile_registry.is_pilot_user("someone-else")

    # Appending a uuid must take effect without a restart; the TTL is the only
    # thing between the write and the next request seeing it.
    roster.write_text(roster.read_text() + "someone-else\n", encoding="utf-8")
    monkeypatch.setattr(profile_registry, "_file_roster_cache", None)
    assert profile_registry.is_pilot_user("someone-else")


def test_missing_roster_file_does_not_admit_everyone(monkeypatch, tmp_path):
    _use(monkeypatch, HERMES_PILOT_ROSTER_FILE=str(tmp_path / "absent.txt"))
    assert not profile_registry.is_pilot_user(UUID_A)


def test_static_list_still_works(monkeypatch):
    _use(monkeypatch, HERMES_PILOT_UUIDS=f"{UUID_A}, other-uuid")
    assert profile_registry.is_pilot_user(UUID_A)
    assert profile_registry.is_pilot_user("other-uuid")
    assert not profile_registry.is_pilot_user("third-uuid")


def test_email_claim_extraction_tolerates_naming(monkeypatch):
    import base64, json

    def token(payload: dict) -> str:
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        return f"Bearer header.{body}.sig"

    assert decode_token_email(token({"email": "A@UGent.be"})) == "a@ugent.be"
    assert decode_token_email(token({"user_email": "b@ugent.be"})) == "b@ugent.be"
    # A uuid in `sub` is not an email and must not be treated as one.
    assert decode_token_email(token({"sub": UUID_A})) is None
    assert decode_token_email(None) is None
