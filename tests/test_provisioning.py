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
    """
    A stand-in hermes-data volume with the config template and SOUL.md.

    Note the name: the template is `config.template.yaml`. Plain `config.yaml`
    on this path is the generated default-profile config, which the AGENT
    rewrites — rendering must not read it.
    """
    (tmp_path / "config.template.yaml").write_text(
        "# a comment the agent would strip\n"
        "mcp_servers:\n  eu-farmbook:\n    env:\n"
        '      EUF_BRIDGE_KEY_FILE: /opt/data/bridge.key\n'
        '      EUF_PROFILE: "__EUF_PROFILE__"\n',
        encoding="utf-8",
    )
    (tmp_path / "SOUL.md").write_text("# EU-FarmBook Farm Assistant\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def configured(monkeypatch, volume):
    s = Settings(
        HERMES_PILOT_UUIDS=f"{UUID_A},{UUID_B}",
        HERMES_DATA_DIR=str(volume), HERMES_MODEL="test-model",
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
    # The rendered config must carry NO credential. It is written to a git-tracked
    # file (hermes-data/config.yaml) and to per-profile configs at mode 0644; the
    # bridge key belongs in bridge.key (0600), which the bridge reads per call.
    assert "bridge-key" not in text
    assert "EUF_BRIDGE_KEY_FILE: /opt/data/bridge.key" in text


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
        HERMES_DATA_DIR=str(volume), HERMES_MODEL="test-model",
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


def test_rotating_the_key_repairs_the_profile_env_and_leaks_nothing_into_config(
    configured, volume, monkeypatch
):
    """
    The key rotates in the profile's .env (0600), never in its config.yaml.

    This test used to assert the opposite — that the rendered config carried the
    key and was rewritten on rotation. That was the bug: the same rendering goes
    into the git-tracked hermes-data/config.yaml.
    """
    profile_registry.resolve_profile(UUID_A)
    config = volume / "profiles" / UUID_A / "config.yaml"
    env = volume / "profiles" / UUID_A / ".env"
    assert "bridge-key" not in config.read_text()
    assert "API_SERVER_KEY=bridge-key" in env.read_text()

    rotated = Settings(HERMES_PILOT_UUIDS=f"{UUID_A},{UUID_B}", HERMES_DATA_DIR=str(volume), HERMES_MODEL="test-model",
                       HERMES_API_KEY="corrected-key", _env_file=None)
    monkeypatch.setattr(provisioning, "S", rotated)
    monkeypatch.setattr(provisioning, "get_settings", lambda: rotated)
    monkeypatch.setattr(profile_registry, "get_settings", lambda: rotated)

    profile_registry.resolve_profile(UUID_A)
    assert "API_SERVER_KEY=corrected-key" in env.read_text()
    assert "corrected-key" not in config.read_text()
    assert oct(env.stat().st_mode)[-3:] == "600"


def test_profile_gets_its_own_env_with_the_scoped_keys(configured, volume, monkeypatch):
    monkeypatch.setattr(provisioning, "S", Settings(
        HERMES_PILOT_UUIDS=UUID_A, HERMES_DATA_DIR=str(volume), HERMES_MODEL="test-model",
        HERMES_API_KEY="bridge-key", LLM_API_KEY="provider-key", _env_file=None))
    provisioning.ensure_profile(UUID_A)

    env = volume / "profiles" / UUID_A / ".env"
    # Hermes resolves a NAMED profile's credentials in that profile's own secret
    # scope and refuses to inherit the listener's key — without this file every
    # request 401s with "no profile-scoped API_SERVER_KEY is configured".
    assert env.is_file()
    text = env.read_text()
    assert "API_SERVER_KEY=bridge-key" in text
    # The name must match `key_env` in the template's providers block,
    # and setting it returns "Unknown provider" AS THE ANSWER, which streams as
    # an empty completion and looks like a broken UI.
    # Must match `key_env` in the providers block of config.yaml — that pairing
    # IS the named-custom-provider contract.
    assert "LLM_API_KEY=provider-key" in text
    assert env.stat().st_mode & 0o777 == 0o600


def test_missing_profile_env_is_repaired(configured, volume):
    provisioning.ensure_profile(UUID_A)
    # Simulate a profile provisioned by an older build, before .env was written.
    (volume / "profiles" / UUID_A / ".env").unlink()
    assert provisioning.ensure_profile(UUID_A) is False
    assert (volume / "profiles" / UUID_A / ".env").is_file()


def test_template_change_reaches_existing_profiles(configured, volume):
    provisioning.ensure_profile(UUID_A)
    rendered = volume / "profiles" / UUID_A / "config.yaml"
    assert "provider: broken" not in rendered.read_text()

    # Edit the shared template the way an operator would — e.g. correcting the
    # provider. Comparing only the keys would leave the profile on the old
    # config forever, which is exactly how "Unknown provider" survived a fix.
    template = volume / "config.template.yaml"
    template.write_text("model:\n  provider: fixed\n" + template.read_text(), encoding="utf-8")

    assert provisioning.ensure_profile(UUID_A) is False
    assert "provider: fixed" in rendered.read_text()


def test_soul_change_reaches_existing_profiles(configured, volume):
    provisioning.ensure_profile(UUID_A)
    (volume / "SOUL.md").write_text("# updated scope contract\n", encoding="utf-8")

    provisioning.ensure_profile(UUID_A)

    # A profile holding an older SOUL.md is a profile running older rules.
    assert (volume / "profiles" / UUID_A / "SOUL.md").read_text() == "# updated scope contract\n"


def test_startup_refresh_updates_every_stale_profile(configured, volume):
    provisioning.ensure_profile(UUID_A)
    provisioning.ensure_profile(UUID_B)

    template = volume / "config.template.yaml"
    template.write_text("model:\n  provider: fixed\n" + template.read_text(), encoding="utf-8")

    # A deploy should reconcile on-disk state immediately, not lazily when each
    # user next happens to send a message.
    assert provisioning.refresh_all_profiles() == 2
    for uid in (UUID_A, UUID_B):
        assert "provider: fixed" in (volume / "profiles" / uid / "config.yaml").read_text()

    # Idempotent: nothing stale, nothing rewritten.
    assert provisioning.refresh_all_profiles() == 0


def test_startup_refresh_survives_a_broken_profile(configured, volume):
    provisioning.ensure_profile(UUID_A)
    (volume / "profiles" / "junk").mkdir()          # not a profile at all
    (volume / "profiles" / UUID_A / ".env").unlink()  # half-provisioned
    assert provisioning.refresh_all_profiles() >= 0   # must not raise


def test_the_model_comes_from_config_not_the_template(configured, volume, monkeypatch):
    from app.config import Settings

    template = volume / "config.template.yaml"
    template.write_text("model:\n  default: __EUF_MODEL__\n" + template.read_text(), encoding="utf-8")

    s = Settings(HERMES_PILOT_UUIDS=UUID_A, HERMES_DATA_DIR=str(volume),
                 HERMES_API_KEY="bridge-key", HERMES_MODEL="magistral-small-latest",
                 _env_file=None)
    monkeypatch.setattr(provisioning, "S", s)
    monkeypatch.setattr(provisioning, "get_settings", lambda: s)

    provisioning.ensure_profile(UUID_A)
    rendered = (volume / "profiles" / UUID_A / "config.yaml").read_text()
    # Comparing models must be an env change plus a restart, not a config edit
    # followed by hand-re-rendering every profile.
    assert "default: magistral-small-latest" in rendered
    assert "__EUF_MODEL__" not in rendered


def test_unset_model_is_refused_rather_than_rendered_empty(monkeypatch, volume):
    """
    An empty HERMES_MODEL used to render `default:` as null, and the agent then
    failed its first completion with a provider error that reads like a bug in
    this service. Refuse where the message names the cause.
    """
    s = Settings(HERMES_PILOT_UUIDS=UUID_A, HERMES_DATA_DIR=str(volume),
                 HERMES_API_KEY="bridge-key", HERMES_MODEL="", _env_file=None)
    monkeypatch.setattr(provisioning, "S", s)
    monkeypatch.setattr(provisioning, "get_settings", lambda: s)

    with pytest.raises(provisioning.ProvisioningError, match="HERMES_MODEL"):
        provisioning.ensure_profile("someone")
    assert not (volume / "profiles" / "someone").exists()


# --- the template the agent cannot eat ------------------------------------
#
# The agent container owns /opt/data/config.yaml: it is the default profile's
# live config, and on startup the agent normalises it, bumping _config_version
# and stripping every comment (124 lines -> 76 on 0.21.1). These pin the split
# that keeps the documented template out of its reach.

def test_rendering_ignores_the_file_the_agent_rewrites(configured, volume):
    """A clobbered config.yaml must not change what profiles are rendered from."""
    (volume / "config.yaml").write_text("_config_version: 42\nmodel:\n  default: junk\n",
                                        encoding="utf-8")
    provisioning.ensure_profile("someone")
    rendered = (volume / "profiles" / "someone" / "config.yaml").read_text(encoding="utf-8")
    assert "junk" not in rendered
    assert "EUF_PROFILE: \"someone\"" in rendered


def test_missing_template_names_the_template_not_config_yaml(configured, volume):
    (volume / "config.template.yaml").unlink()
    with pytest.raises(provisioning.ProvisioningError, match="config.template.yaml"):
        provisioning.ensure_profile("someone")


def test_default_config_is_generated_from_the_template_without_comments(configured, volume):
    """
    The default profile is what a request with no /p/<profile>/ prefix lands on,
    so it must carry our hardening rather than the agent's defaults. Comments are
    stripped so the agent's own normalisation leaves the tracked file alone.
    """
    assert provisioning.write_default_config() is True
    text = (volume / "config.yaml").read_text(encoding="utf-8")
    assert "__EUF_" not in text
    assert 'EUF_PROFILE: "default"' in text
    assert not [line for line in text.splitlines() if line.lstrip().startswith("#")]
    # Idempotent: a second startup must not rewrite an already-current file.
    assert provisioning.write_default_config() is False


def test_default_config_is_left_alone_when_the_model_is_unset(monkeypatch, volume):
    s = Settings(HERMES_DATA_DIR=str(volume), HERMES_API_KEY="bridge-key",
                 HERMES_MODEL="", _env_file=None)
    monkeypatch.setattr(provisioning, "S", s)
    monkeypatch.setattr(provisioning, "get_settings", lambda: s)
    (volume / "config.yaml").write_text("whatever the agent wrote\n", encoding="utf-8")

    assert provisioning.write_default_config() is False
    # Never raises at startup, and never truncates what is already there.
    assert (volume / "config.yaml").read_text(encoding="utf-8") == "whatever the agent wrote\n"


def test_comment_stripping_keeps_a_hash_inside_a_value():
    stripped = provisioning._strip_comments(
        "# leading comment\n\n\nbase_url: https://x/v1\nnote: \"a # inside a value\"\n"
    )
    assert stripped == 'base_url: https://x/v1\nnote: "a # inside a value"\n'
