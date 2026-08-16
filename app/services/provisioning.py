# app/services/provisioning.py
"""
Create a pilot user's Hermes profile on first use.

**Why this can be automatic.** A Hermes profile is a directory under
`<data>/profiles/<id>/`, and the gateway resolves `/p/<profile>/` by calling
`profiles_to_serve()` — "intentionally lightweight (a directory scan + name
validation only)" — on **every request** (`api_server._resolve_request_profile`).
So a directory created after the gateway started is served on the next request,
with no restart and no `hermes profile create`. Writing the directory is enough.

That is why the adapter does not need a docker socket or a shell into the agent
container: it shares the data volume and writes a directory. The earlier design
assumed provisioning required the CLI, which would have meant an operator running
a script before any new user could chat.

**Under open access this runs for every new user**, which is the intent: someone
arrives, chats, and their agent exists. The bounds that keep that survivable are
elsewhere — `rate_limit` caps turns per user, and `MAX_PROFILES` (checked below)
caps how many agents can ever exist. Provisioning itself stays deliberately
dumb: it writes a directory, and refuses rather than improvising if it cannot.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from app.config import get_settings

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.provisioning")

# Same skeleton hermes_cli.profiles bootstraps into a new profile
# (`_PROFILE_DIRS`). Hermes recreates what it needs, but starting from the same
# shape avoids surprising a code path that assumes one of these exists.
_PROFILE_DIRS = (
    "memories", "sessions", "skills", "skins", "logs", "plans", "workspace", "cron", "home",
)


class ProvisioningError(Exception):
    """The profile could not be created; the caller must not fall back to a shared one."""


def _data_dir() -> Path:
    return Path(S.HERMES_DATA_DIR)


def profile_dir(profile: str) -> Path:
    return _data_dir() / "profiles" / profile


def is_provisioned(profile: str) -> bool:
    return (profile_dir(profile) / "config.yaml").is_file()


def _render_profile_env() -> str:
    """
    Secrets for one profile, written to `<profile>/.env`.

    Required, not optional. Under multiplexing Hermes resolves a named profile's
    credentials **inside that profile's own secret scope** and deliberately
    refuses to borrow the listener's — `_expected_api_key()` in
    `gateway/platforms/api_server.py`: "Named profiles must fail closed rather
    than inherit the listener owner's key." A profile without this file answers
    every request with:

        API server rejected request for profile '<id>':
        no profile-scoped API_SERVER_KEY is configured

    The provider key is scoped the same way, so it has to be here too or the
    agent authenticates and then cannot reach the model.
    """
    lines = [
        "# Written by farm_assistant_hermes on provisioning. Do not edit by hand:",
        "# it is regenerated whenever the adapter's keys change.",
        f"API_SERVER_KEY={S.HERMES_API_KEY}",
    ]
    if S.MISTRAL_API_KEY:
        # `provider: custom` in config.yaml reads OPENAI_* — Mistral is not a
        # Hermes provider id, it is an OpenAI-compatible endpoint reached this
        # way. Same seam a self-hosted vLLM would use.
        lines.append(f"OPENAI_BASE_URL={S.MISTRAL_API_URL}/v1")
        lines.append(f"OPENAI_API_KEY={S.MISTRAL_API_KEY}")
    else:
        logger.warning(
            "MISTRAL_API_KEY is unset — provisioned profiles will have no provider "
            "credential and the agent will fail on its first completion."
        )
    return "\n".join(lines) + "\n"


def _write_profile_env(directory: Path) -> None:
    env_path = directory / ".env"
    env_path.write_text(_render_profile_env(), encoding="utf-8")
    # Same-uid-only: the agent runs as the adapter's uid, and nothing else on
    # the volume needs to read a profile's credentials.
    env_path.chmod(0o600)


def _config_is_current(profile: str) -> bool:
    """
    Does this profile's config still match what we would write today?

    The bridge key is the part that rots: EUF_BRIDGE_KEY is baked into each
    profile at provisioning time, so rotating HERMES_API_KEY (or correcting a
    mismatch with API_SERVER_KEY) silently invalidates every existing profile —
    the agent keeps answering, but every tool call 401s against the adapter, and
    the only symptom is an assistant that has mysteriously stopped searching.
    """
    directory = profile_dir(profile)
    try:
        current = (directory / "config.yaml").read_text(encoding="utf-8")
        env = (directory / ".env").read_text(encoding="utf-8")
    except OSError:
        return False
    fresh = (
        f'EUF_BRIDGE_KEY: "{S.HERMES_API_KEY}"' in current
        and f"API_SERVER_KEY={S.HERMES_API_KEY}" in env
    )
    if fresh and S.MISTRAL_API_KEY:
        fresh = f"OPENAI_API_KEY={S.MISTRAL_API_KEY}" in env
    return fresh


def _render_config(profile: str) -> str:
    """
    Fill the shared config template for one profile.

    EUF_PROFILE must be this profile's own id: it is how a tool call coming back
    from the agent is matched to the turn that is in flight. Two profiles sharing
    it would cross user turns.
    """
    template = (_data_dir() / "config.yaml").read_text(encoding="utf-8")
    rendered = (
        template
        .replace("__EUF_PROFILE__", profile)
        .replace("__EUF_BRIDGE_KEY__", S.HERMES_API_KEY)
    )
    if "__EUF_" in rendered:
        raise ProvisioningError("config.yaml template has unsubstituted placeholders")
    return rendered


def profile_count() -> int:
    root = _data_dir() / "profiles"
    if not root.is_dir():
        return 0
    return sum(1 for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith("."))


def ensure_profile(profile: str) -> bool:
    """
    Make sure `profile` exists on the shared volume. Returns True if it was
    created by this call, False if it already existed.

    Concurrency: two simultaneous first turns for the same user race here. The
    directory is built under a temp name and moved into place, and an existing
    destination is treated as success — whoever lost the race gets a profile
    that is just as valid as the one they were writing.
    """
    target = profile_dir(profile)
    if is_provisioned(profile):
        if not _config_is_current(profile):
            # Repair in place rather than refusing or rebuilding: the profile's
            # sessions and agent state are fine, only the rendered config is stale.
            try:
                (target / "config.yaml").write_text(_render_config(profile), encoding="utf-8")
                _write_profile_env(target)
                logger.warning("Refreshed stale config/.env for profile %s (keys changed)", profile)
            except OSError as e:
                raise ProvisioningError(f"could not refresh config for {profile}: {e}") from e
        return False

    # Disk and blast-radius ceiling under open access. Existing users are
    # unaffected — only the creation of a NEW agent is refused — so hitting this
    # degrades enrolment rather than breaking the service.
    max_profiles = S.MAX_PROFILES
    if max_profiles and profile_count() >= max_profiles:
        logger.error(
            "Refusing to provision %s: MAX_PROFILES=%s reached", profile, max_profiles
        )
        raise ProvisioningError(f"profile limit reached ({max_profiles})")

    data_dir = _data_dir()
    if not (data_dir / "config.yaml").is_file():
        raise ProvisioningError(
            f"No config.yaml template at {data_dir} — is the hermes-data volume mounted?"
        )

    config_text = _render_config(profile)
    soul_path = data_dir / "SOUL.md"

    profiles_root = data_dir / "profiles"
    try:
        profiles_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{profile}.", dir=profiles_root))

        for name in _PROFILE_DIRS:
            (staging / name).mkdir(parents=True, exist_ok=True)

        (staging / "config.yaml").write_text(config_text, encoding="utf-8")
        _write_profile_env(staging)
        if soul_path.is_file():
            shutil.copyfile(soul_path, staging / "SOUL.md")
        else:
            # Without SOUL.md the agent loses its scope contract. scope.py still
            # restates the rules per turn, so this degrades rather than opens the
            # agent up — but it is a misconfiguration and should be loud.
            logger.error("SOUL.md missing at %s; profile %s created without it", soul_path, profile)

        try:
            os.rename(staging, target)
        except OSError:
            # Lost the race (or a stale dir exists). If the winner produced a
            # usable profile we are done; otherwise this is a real failure.
            shutil.rmtree(staging, ignore_errors=True)
            if is_provisioned(profile):
                return False
            raise
    except ProvisioningError:
        raise
    except OSError as e:
        logger.error("Could not provision profile %s: %s", profile, e)
        raise ProvisioningError(str(e)) from e

    logger.info("Provisioned Hermes profile %s", profile)
    return True


def profile_name_for(user_uuid: str, override: Optional[str] = None) -> str:
    """
    Deterministic profile id for a user.

    Defaults to the uuid itself: Hermes' profile id rule is
    `^[a-z0-9][a-z0-9_-]{0,63}$`, which a lowercased uuid satisfies, and using it
    directly means the mapping needs no state and cannot drift. An explicit
    override from HERMES_PROFILE_MAP wins, for the handful of profiles that were
    seeded by hand with readable names.
    """
    return override or user_uuid.strip().lower()
