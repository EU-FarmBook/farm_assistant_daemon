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

**Two config files, and only one is source.** `config.template.yaml` is the
documented template this module renders from; the agent never reads it.
`config.yaml` is GENERATED — it is the default profile's live config, and the
agent container normalises it on startup, stripping every comment. Keeping the
template on a separate path is what stops that normalisation from eating the
documentation; `write_default_config()` below keeps the generated file in step.

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

# The documented source, and the generated file the agent owns. See the module
# docstring — the split exists because the agent rewrites the latter.
_TEMPLATE_NAME = "config.template.yaml"
_DEFAULT_CONFIG_NAME = "config.yaml"

# The profile id the default config is rendered for. It never serves a turn (the
# adapter always addresses /p/<profile>/), so a tool call arriving tagged with it
# finds no active turn and is refused — which is the intended outcome.
_DEFAULT_PROFILE = "default"


class ProvisioningError(Exception):
    """The profile could not be created; the caller must not fall back to a shared one."""


def _data_dir() -> Path:
    return Path(S.HERMES_DATA_DIR)


def profile_dir(profile: str) -> Path:
    return _data_dir() / "profiles" / profile


def template_path() -> Path:
    return _data_dir() / _TEMPLATE_NAME


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
    if S.LLM_API_KEY:
        # The name here must match `key_env` in the `providers` block of
        # config.yaml — that is the whole contract for a named custom provider:
        # config says which variable holds the key, this writes that variable.
        # Get it wrong and the agent authenticates against :8642 fine, then
        # fails its first completion.
        lines.append(f"LLM_API_KEY={S.LLM_API_KEY}")
    else:
        logger.warning(
            "LLM_API_KEY is unset — provisioned profiles will have no provider "
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

    Compares the WHOLE rendered template, not just the credentials. A profile's
    config.yaml is a derived artifact — every edit to hermes-data/config.yaml
    (the provider, the toolsets, the memory switches, the MCP block) has to
    reach existing profiles or they keep running yesterday's configuration
    while the template says otherwise. That is not hypothetical: fixing
    a `provider:` fix in the template changed nothing for the one
    profile that already existed, because only the keys were being compared.

    Same for SOUL.md: it carries the scope contract, and a profile silently
    holding an older copy is a profile with older rules.
    """
    directory = profile_dir(profile)
    try:
        current = (directory / "config.yaml").read_text(encoding="utf-8")
        env = (directory / ".env").read_text(encoding="utf-8")
    except OSError:
        return False

    if current != _render_config(profile) or env != _render_profile_env():
        return False

    soul = _data_dir() / "SOUL.md"
    if soul.is_file():
        try:
            if (directory / "SOUL.md").read_text(encoding="utf-8") != soul.read_text(encoding="utf-8"):
                return False
        except OSError:
            return False

    return True


def _render_config(profile: str) -> str:
    """
    Fill the shared config template for one profile.

    EUF_PROFILE must be this profile's own id: it is how a tool call coming back
    from the agent is matched to the turn that is in flight. Two profiles sharing
    it would cross user turns.
    """
    # An empty model id renders `default:` as null and the agent fails at its
    # first completion with a provider error that reads like a bug in here.
    # Refuse at provisioning instead, where the message names the cause.
    if not S.HERMES_MODEL.strip():
        raise ProvisioningError(
            "HERMES_MODEL is unset — set it to a model the provider serves "
            "(list them: curl -s $LLM_API_URL/v1/models -H \"Authorization: Bearer $LLM_API_KEY\")"
        )

    # NOTE what is NOT substituted: the bridge key. The rendered config goes to
    # a git-tracked file (hermes-data/config.yaml) and to per-profile configs at
    # mode 0644, so it must stay credential-free. The template points the bridge
    # at /opt/data/bridge.key (0600, written by write_bridge_key) instead.
    template = template_path().read_text(encoding="utf-8")
    rendered = (
        template
        .replace("__EUF_PROFILE__", profile)
        .replace("__EUF_MODEL__", S.HERMES_MODEL)
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
                soul = _data_dir() / "SOUL.md"
                if soul.is_file():
                    shutil.copyfile(soul, target / "SOUL.md")
                logger.warning(
                    "Refreshed profile %s from the current template (config/.env/SOUL)", profile
                )
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
    if not template_path().is_file():
        raise ProvisioningError(
            f"No {_TEMPLATE_NAME} at {data_dir} — is the hermes-data volume mounted? "
            "(config.yaml is the generated default-profile config, not the template.)"
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


def _strip_comments(text: str) -> str:
    """
    Drop whole-line comments and collapse the blank runs they leave behind.

    Only lines that are entirely a comment: a `#` inside a value is left alone.
    The point is to hand the agent a file it has nothing left to normalise, so
    its startup rewrite stops showing up as a 48-line diff on a tracked file.
    """
    kept: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if not line.strip() and (not kept or not kept[-1].strip()):
            continue
        kept.append(line)
    return "\n".join(kept).strip() + "\n"


def write_default_config() -> bool:
    """
    Render the template into the default profile's live `config.yaml`.

    Returns True if the file was written, False if it was already current.

    This is the one place that file should ever be written by us. It matters
    because the DEFAULT profile is what a request without a /p/<profile>/ prefix
    lands on, and a config the agent generated for itself would come with the
    built-in memory on and the full platform toolset — terminal included — which
    is the opposite of every other decision in this deployment.

    Comments are stripped: the agent normalises this file on startup anyway, so
    handing it a comment-free rendering keeps a tracked file from churning.

    Never raises. A startup that cannot write this still serves; the profiles
    that actually answer users are rendered separately.
    """
    target = _data_dir() / _DEFAULT_CONFIG_NAME
    try:
        if not S.HERMES_MODEL.strip():
            logger.warning(
                "HERMES_MODEL is unset — leaving %s as it is. The default profile keeps "
                "whatever config it already has, which on a fresh volume is the agent's "
                "own (memory on, full toolset).", target,
            )
            return False

        rendered = _strip_comments(_render_config(_DEFAULT_PROFILE))
        if target.is_file() and target.read_text(encoding="utf-8") == rendered:
            return False
        target.write_text(rendered, encoding="utf-8")
        logger.info("Wrote %s from %s", target, _TEMPLATE_NAME)
        return True
    except (OSError, ProvisioningError) as e:
        logger.error("Could not write the default profile config at %s: %s", target, e)
        return False


def write_bridge_key() -> None:
    """
    Publish the current bridge key where the MCP bridge reads it.

    One file for all profiles, rewritten at startup, so rotating HERMES_API_KEY
    takes effect on the next tool call rather than the next respawn of every
    agent subprocess.
    """
    if not S.HERMES_API_KEY:
        logger.warning("HERMES_API_KEY is empty — the MCP bridge cannot authenticate.")
        return
    path = _data_dir() / "bridge.key"
    try:
        path.write_text(S.HERMES_API_KEY, encoding="utf-8")
        path.chmod(0o600)
    except OSError as e:
        logger.error("Could not write %s: %s", path, e)


def refresh_all_profiles() -> int:
    """
    Re-render every existing profile from the current template. Returns the
    number refreshed.

    Called at startup so a deploy reconciles on-disk state immediately instead
    of lazily, one user's first turn at a time. Without it, `git pull` +
    `compose up` leaves every profile on the previous template until its owner
    happens to send a message — which reads, correctly, as "the fix did not
    work".

    Never raises: a profile that cannot be refreshed must not stop the service
    from starting. It will be retried on that user's next turn.
    """
    root = _data_dir() / "profiles"
    if not root.is_dir():
        return 0

    refreshed = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        try:
            if not is_provisioned(entry.name):
                continue
            if _config_is_current(entry.name):
                continue
            ensure_profile(entry.name)
            refreshed += 1
        except Exception as e:  # noqa: BLE001 - startup must survive anything here
            logger.error("Could not refresh profile %s: %s", entry.name, e)
    return refreshed


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
