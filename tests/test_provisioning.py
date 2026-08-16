"""
Tests for automatic profile provisioning.

The property that matters: a pilot user's first turn creates THEIR OWN profile,
and a non-pilot user creates nothing and is refused. Auto-provisioning must not
become a way for any authenticated account to get an agent.
"""

import pytest

from app.config import Settings
from app.services import profile_registry, provisioning
from app.services.profile_registry import ProfileNotProvisioned

UUID_A = "45b75f62-3fa3-4b18-8593-1411f110a98e"
UUID_B = "9d1f0c31-1111-4c22-9aaa-2b3c4d5e6f70"
UUID_OUTSIDE = "00000000-dead-4bee-8000-000000000000"


@pytest.fixture
def volume(tmp_path):
    """A stand-in hermes-data volume with the config template and SOUL.md."""
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  eu-farmbook:\n    env:\n"
        '      EUF_BRIDGE_KEY: "__EUF_BRIDGE_KEY__"\n'
        '      EUF_PROFILE: "__EUF_PROFILE__"\n',
        encoding="utf-8",
    )
    (tmp_path / "SOUL.md").write_text("# EU-FarmBook Farm Assistant\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def configured(monkeypatch, volume):
    s = Settings(
        HERMES_PILOT_UUIDS=f"{UUID_A},{UUID_B}",
        HERMES_DATA_DIR=str(volume),
        HERMES_API_KEY="bridge-key",
        _env_file=None,
    )
    monkeypatch.setattr(provisioning, "S", s)
    monkeypatch.setattr(provisioning, "get_settings", lambda: s)
    monkeypatch.setattr(profile_registry, "get_settings", lambda: s)
    return s


def test_first_turn_creates_the_users_own_profile(configured, volume):
    profile = profile_registry.resolve_profile(UUID_A)
    assert profile == UUID_A
    assert (volume / "profiles" / UUID_A / "config.yaml").is_file()
    assert (volume / "profiles" / UUID_A / "SOUL.md").is_file()
    # The skeleton Hermes bootstraps into every profile.
    assert (volume / "profiles" / UUID_A / "memories").is_dir()
    assert (volume / "profiles" / UUID_A / "sessions").is_dir()


def test_each_user_gets_a_separate_profile(configured, volume):
    a = profile_registry.resolve_profile(UUID_A)
    b = profile_registry.resolve_profile(UUID_B)
    assert a != b
    assert (volume / "profiles" / a).is_dir()
    assert (volume / "profiles" / b).is_dir()


def test_placeholders_are_substituted_per_profile(configured, volume):
    profile_registry.resolve_profile(UUID_A)
    text = (volume / "profiles" / UUID_A / "config.yaml").read_text()
    assert "__EUF_" not in text
    # EUF_PROFILE is how a tool call is matched back to an in-flight turn; if two
    # profiles shared it, users' turns would cross.
    assert f'EUF_PROFILE: "{UUID_A}"' in text
    assert 'EUF_BRIDGE_KEY: "bridge-key"' in text


def test_second_turn_does_not_recreate(configured):
    assert provisioning.ensure_profile(UUID_A) is True
    assert provisioning.ensure_profile(UUID_A) is False


def test_non_pilot_user_is_refused_and_provisions_nothing(configured, volume):
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile(UUID_OUTSIDE)
    assert not (volume / "profiles" / UUID_OUTSIDE).exists()


def test_anonymous_is_refused(configured):
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile(None)


def test_naming_override_is_honoured_and_implies_roster(monkeypatch, volume):
    s = Settings(
        HERMES_PROFILE_MAP=f"{UUID_OUTSIDE}:alice",
        HERMES_DATA_DIR=str(volume),
        HERMES_API_KEY="bridge-key",
        _env_file=None,
    )
    monkeypatch.setattr(provisioning, "S", s)
    monkeypatch.setattr(provisioning, "get_settings", lambda: s)
    monkeypatch.setattr(profile_registry, "get_settings", lambda: s)

    assert profile_registry.resolve_profile(UUID_OUTSIDE) == "alice"
    assert (volume / "profiles" / "alice" / "config.yaml").is_file()


def test_provisioning_failure_refuses_rather_than_sharing(configured, monkeypatch):
    def boom(_profile):
        raise provisioning.ProvisioningError("disk on fire")

    monkeypatch.setattr(provisioning, "ensure_profile", boom)
    # The dangerous fallback would be "couldn't make you one, use the default" —
    # that would put this user's conversation inside a shared agent.
    with pytest.raises(ProfileNotProvisioned):
        profile_registry.resolve_profile(UUID_A)


def test_missing_template_is_a_hard_failure(monkeypatch, tmp_path):
    s = Settings(
        HERMES_PILOT_UUIDS=UUID_A,
        HERMES_DATA_DIR=str(tmp_path),  # no config.yaml — volume not mounted
        _env_file=None,
    )
    monkeypatch.setattr(provisioning, "S", s)
    monkeypatch.setattr(provisioning, "get_settings", lambda: s)
    with pytest.raises(provisioning.ProvisioningError):
        provisioning.ensure_profile(UUID_A)


def test_stale_bridge_key_is_repaired_in_place(configured, volume, monkeypatch):
    profile_registry.resolve_profile(UUID_A)
    config = volume / "profiles" / UUID_A / "config.yaml"
    assert 'EUF_BRIDGE_KEY: "bridge-key"' in config.read_text()

    # Mark the profile as having some state, so we can prove it survives.
    (volume / "profiles" / UUID_A / "sessions" / "keep.json").write_text("{}", encoding="utf-8")

    # The operator corrects HERMES_API_KEY to match API_SERVER_KEY.
    rotated = Settings(HERMES_PILOT_UUIDS=f"{UUID_A},{UUID_B}", HERMES_DATA_DIR=str(volume),
                       HERMES_API_KEY="corrected-key", _env_file=None)
    monkeypatch.setattr(provisioning, "S", rotated)
    monkeypatch.setattr(provisioning, "get_settings", lambda: rotated)
    monkeypatch.setattr(profile_registry, "get_settings", lambda: rotated)

    profile_registry.resolve_profile(UUID_A)

    # Config repaired, state untouched — otherwise every tool call would 401
    # against the adapter with no visible cause.
    assert 'EUF_BRIDGE_KEY: "corrected-key"' in config.read_text()
    assert (volume / "profiles" / UUID_A / "sessions" / "keep.json").is_file()


def test_profile_gets_its_own_env_with_the_scoped_keys(configured, volume, monkeypatch):
    monkeypatch.setattr(provisioning, "S", Settings(
        HERMES_PILOT_UUIDS=UUID_A, HERMES_DATA_DIR=str(volume),
        HERMES_API_KEY="bridge-key", MISTRAL_API_KEY="mistral-key", _env_file=None))
    provisioning.ensure_profile(UUID_A)

    env = volume / "profiles" / UUID_A / ".env"
    # Hermes resolves a NAMED profile's credentials in that profile's own secret
    # scope and refuses to inherit the listener's key — without this file every
    # request 401s with "no profile-scoped API_SERVER_KEY is configured".
    assert env.is_file()
    text = env.read_text()
    assert "API_SERVER_KEY=bridge-key" in text
    # `provider: custom` reads OPENAI_* — "mistral" is not a Hermes provider id,
    # and setting it returns "Unknown provider" AS THE ANSWER, which streams as
    # an empty completion and looks like a broken UI.
    assert "OPENAI_API_KEY=mistral-key" in text
    assert "OPENAI_BASE_URL=https://api.mistral.ai/v1" in text
    assert env.stat().st_mode & 0o777 == 0o600


def test_missing_profile_env_is_repaired(configured, volume):
    provisioning.ensure_profile(UUID_A)
    # Simulate a profile provisioned by an older build, before .env was written.
    (volume / "profiles" / UUID_A / ".env").unlink()
    assert provisioning.ensure_profile(UUID_A) is False
    assert (volume / "profiles" / UUID_A / ".env").is_file()
