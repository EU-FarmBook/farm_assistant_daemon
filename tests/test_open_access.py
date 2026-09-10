"""
Open access: any authenticated user gets their own agent, with bounds.

The point of these is that "open" removes a bound on WHO, and nothing else —
identity is still verified, users are still isolated from each other, and spend
is still capped per user.
"""

import pytest

from app.config import Settings
from app.services import profile_registry, provisioning, rate_limit
from app.services.profile_registry import ProfileNotProvisioned

UUID_A = "45b75f62-3fa3-4b18-8593-1411f110a98e"
UUID_B = "9d1f0c31-1111-4c22-9aaa-2b3c4d5e6f70"


@pytest.fixture
def volume(tmp_path):
    (tmp_path / "config.template.yaml").write_text(
        'env:\n  EUF_BRIDGE_KEY_FILE: /opt/data/bridge.key\n  EUF_PROFILE: "__EUF_PROFILE__"\n',
        encoding="utf-8",
    )
    (tmp_path / "SOUL.md").write_text("# scope contract\n", encoding="utf-8")
    return tmp_path


def _open(monkeypatch, volume, **kwargs):
    s = Settings(HERMES_OPEN_ACCESS=True, HERMES_DATA_DIR=str(volume), HERMES_MODEL="test-model",
                 HERMES_API_KEY="k", _env_file=None, **kwargs)
    monkeypatch.setattr(provisioning, "S", s)
    monkeypatch.setattr(provisioning, "get_settings", lambda: s)
    monkeypatch.setattr(profile_registry, "get_settings", lambda: s)
    monkeypatch.setattr(rate_limit, "get_settings", lambda: s)
    rate_limit.reset()
    return s


def test_any_authenticated_user_is_admitted(monkeypatch, volume):
    _open(monkeypatch, volume)
    # No roster configured at all — that is the whole point.
    assert profile_registry.resolve_profile(UUID_A) == UUID_A
    assert profile_registry.resolve_profile(UUID_B) == UUID_B


def test_open_access_still_requires_authentication(monkeypatch, volume):
    _open(monkeypatch, volume)
    # Open means "no roster", never "no login". An unauthenticated request has
    # no verified uuid, so there is no agent to route it to.
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile(None)
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile("")


def test_users_remain_isolated_under_open_access(monkeypatch, volume):
    _open(monkeypatch, volume)
    a = profile_registry.resolve_profile(UUID_A)
    b = profile_registry.resolve_profile(UUID_B)
    assert a != b
    assert (volume / "profiles" / a / "config.yaml").read_text() != (
        volume / "profiles" / b / "config.yaml"
    ).read_text()


def test_max_profiles_refuses_new_agents_but_not_existing_ones(monkeypatch, volume):
    _open(monkeypatch, volume, MAX_PROFILES=1)
    assert profile_registry.resolve_profile(UUID_A) == UUID_A
    # A second NEW user is refused...
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile(UUID_B)
    # ...while the existing one keeps working.
    assert profile_registry.resolve_profile(UUID_A) == UUID_A


# --- the limiter ---------------------------------------------------------

def test_per_minute_limit_refuses_before_spending(monkeypatch, volume):
    _open(monkeypatch, volume, RATE_LIMIT_TURNS_PER_MIN=3, RATE_LIMIT_TURNS_PER_DAY=0)
    for _ in range(3):
        rate_limit.check_and_record(UUID_A)
    with pytest.raises(rate_limit.RateLimited) as exc:
        rate_limit.check_and_record(UUID_A)
    assert exc.value.scope == "minute"
    assert exc.value.retry_after_seconds >= 1


def test_daily_limit(monkeypatch, volume):
    _open(monkeypatch, volume, RATE_LIMIT_TURNS_PER_MIN=0, RATE_LIMIT_TURNS_PER_DAY=2)
    rate_limit.check_and_record(UUID_A)
    rate_limit.check_and_record(UUID_A)
    with pytest.raises(rate_limit.RateLimited) as exc:
        rate_limit.check_and_record(UUID_A)
    assert exc.value.scope == "day"


def test_limits_are_per_user_not_global(monkeypatch, volume):
    _open(monkeypatch, volume, RATE_LIMIT_TURNS_PER_MIN=1)
    rate_limit.check_and_record(UUID_A)
    with pytest.raises(rate_limit.RateLimited):
        rate_limit.check_and_record(UUID_A)
    # One noisy user must not lock everyone else out.
    rate_limit.check_and_record(UUID_B)


def test_limiter_can_be_disabled(monkeypatch, volume):
    _open(monkeypatch, volume, RATE_LIMIT_ENABLED=False, RATE_LIMIT_TURNS_PER_MIN=1)
    for _ in range(20):
        rate_limit.check_and_record(UUID_A)
