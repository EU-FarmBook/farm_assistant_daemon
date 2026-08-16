# app/services/profile_registry.py
"""
uuid -> Hermes profile, and the pilot allowlist.

Two separate questions, deliberately answered by different mechanisms:

**Is this user allowed?** With `HERMES_OPEN_ACCESS`, any authenticated user is —
they arrive, they chat, their agent is created on the first message, and no
operator touches anything. Otherwise a roster decides: an email domain
(`HERMES_PILOT_EMAIL_DOMAINS`), a hot-reloaded file (`HERMES_PILOT_ROSTER_FILE`),
or the static `HERMES_PILOT_UUIDS`; with none configured nobody is allowed, so
the roster path fails closed rather than open.

Open or not, identity is still VERIFIED — `user_uuid` comes from a token
introspected against Django. Open access removes a bound on *who*, never the
check on *whether they are who they say*. There is also no "default profile"
fallback and there must never be one, since Hermes' agent state is per profile
home.

**Does their agent exist yet?** Not a question the operator should have to
answer. The profile is created on first use by `provisioning.ensure_profile()`,
because the gateway discovers profiles by directory scan on every request. See
that module for why this needs no docker socket.

`HERMES_PROFILE_MAP` is still honoured for profiles seeded by hand with readable
names (`uuid:alice`); it is now a naming override, not the roster.
"""

import logging
import re
import time
from typing import Optional

from app.config import get_settings
from app.services import provisioning

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.profiles")

# Identical to Hermes' own `_PROFILE_ID_RE` (hermes_cli/profiles.py). A profile
# id becomes a URL path segment and a directory name, so anything outside this
# charset must never reach either.
_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Roster-file cache. Short TTL: long enough that a busy stream is not stat-ing a
# file per turn, short enough that adding a user feels immediate.
_ROSTER_TTL_SECONDS = 30.0
_file_roster_cache: Optional[set[str]] = None
_file_roster_read_at: float = 0.0


class ProfileNotProvisioned(Exception):
    """The caller is authenticated but not part of the pilot, or provisioning failed."""

    def __init__(self, user_uuid: str, reason: str = "not on the pilot roster"):
        self.user_uuid = user_uuid
        self.reason = reason
        super().__init__(f"No Hermes profile for user {user_uuid}: {reason}")


def is_valid_profile_name(name: str) -> bool:
    return bool(name) and bool(_PROFILE_RE.match(name))


def _static_roster() -> set[str]:
    settings = get_settings()
    listed = {
        entry.strip()
        for entry in (settings.HERMES_PILOT_UUIDS or "").split(",")
        if entry.strip()
    }
    # A uuid named in the naming-override map is implicitly on the roster —
    # otherwise hand-seeded profiles would be unreachable.
    return listed | set(settings.profile_map())


def _file_roster() -> set[str]:
    """
    Uuids from HERMES_PILOT_ROSTER_FILE, re-read on a short TTL.

    The point is operational: adding someone becomes `echo <uuid> >> roster.txt`
    on the server, with no container restart and no redeploy. Blank lines and
    `#` comments are ignored so the file can carry names next to the uuids.
    """
    settings = get_settings()
    path = (settings.HERMES_PILOT_ROSTER_FILE or "").strip()
    if not path:
        return set()

    global _file_roster_cache, _file_roster_read_at
    now = time.monotonic()
    if _file_roster_cache is not None and (now - _file_roster_read_at) < _ROSTER_TTL_SECONDS:
        return _file_roster_cache

    entries: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                entry = line.split("#", 1)[0].strip()
                if entry:
                    entries.add(entry)
    except FileNotFoundError:
        logger.warning("Roster file %s does not exist yet", path)
    except OSError as e:
        # Keep the last good roster rather than locking everyone out on a
        # transient read error.
        logger.error("Could not read roster file %s: %s", path, e)
        return _file_roster_cache or set()

    _file_roster_cache, _file_roster_read_at = entries, now
    return entries


def _allowed_domains() -> set[str]:
    return {
        d.strip().lower().lstrip("@")
        for d in (get_settings().HERMES_PILOT_EMAIL_DOMAINS or "").split(",")
        if d.strip()
    }


def is_pilot_user(user_uuid: Optional[str], email: Optional[str] = None) -> bool:
    """
    Membership test.

    `email` must come from a token that has already been VERIFIED by
    auth_service — the claims of a verified token are the issuer's, so acting on
    them is safe; acting on an unverified one would let a caller pick their own
    domain.
    """
    if not user_uuid:
        return False

    if get_settings().HERMES_OPEN_ACCESS:
        return True

    domains = _allowed_domains()
    if domains and email:
        domain = email.rsplit("@", 1)[-1].lower()
        if domain in domains:
            return True

    return user_uuid in (_static_roster() | _file_roster())


def resolve_profile(
    user_uuid: Optional[str],
    *,
    email: Optional[str] = None,
    provision: bool = True,
) -> str:
    """
    Return the Hermes profile for a VERIFIED user uuid, creating it if needed.

    The uuid must come from auth_service.resolve_user_uuid() — never from a
    request header or body. The whole isolation guarantee reduces to that:
    whoever controls this string controls which agent is addressed.
    """
    if not user_uuid:
        raise ProfileNotProvisioned("<anonymous>", "not authenticated")

    if not is_pilot_user(user_uuid, email):
        # Logged at info with the uuid so an operator can add someone by having
        # them click the link once and copying the uuid out of the logs — the
        # uuid is otherwise awkward to find.
        logger.info("Rejecting chat for uuid=%s (not on roster)", user_uuid)
        raise ProfileNotProvisioned(user_uuid)

    profile = provisioning.profile_name_for(
        user_uuid, override=get_settings().profile_map().get(user_uuid)
    )

    if not is_valid_profile_name(profile):
        # Misconfiguration, not user input — fail loudly rather than sending a
        # traversal-shaped segment to Hermes or to the filesystem.
        logger.error("Profile %r for uuid=%s is not a valid profile name", profile, user_uuid)
        raise ProfileNotProvisioned(user_uuid, "invalid profile name")

    if provision:
        try:
            if provisioning.ensure_profile(profile):
                logger.info("First turn for uuid=%s — provisioned profile %s", user_uuid, profile)
        except provisioning.ProvisioningError as e:
            # Refuse rather than degrade. Falling back to the default profile
            # here would put this user's conversation in a shared agent.
            raise ProfileNotProvisioned(user_uuid, f"provisioning failed: {e}") from e

    return profile


def pilot_size() -> int:
    """Explicitly-listed uuids. A domain rule admits users not counted here."""
    return len(_static_roster() | _file_roster())


def gate_description() -> str:
    """One-line summary of how access is gated, for the startup log."""
    settings = get_settings()
    if settings.HERMES_OPEN_ACCESS:
        limiter = (
            f"limit={settings.RATE_LIMIT_TURNS_PER_MIN}/min,"
            f"{settings.RATE_LIMIT_TURNS_PER_DAY}/day"
            if settings.RATE_LIMIT_ENABLED else "LIMITER DISABLED"
        )
        cap = f"max_profiles={settings.MAX_PROFILES}" if settings.MAX_PROFILES else "profiles=unlimited"
        return f"OPEN to all authenticated users ({limiter}, {cap})"

    parts = []
    if _allowed_domains():
        parts.append("domains=" + ",".join(sorted(_allowed_domains())))
    if get_settings().HERMES_PILOT_ROSTER_FILE:
        parts.append(f"roster_file={get_settings().HERMES_PILOT_ROSTER_FILE}")
    if _static_roster():
        parts.append(f"uuids={len(_static_roster())}")
    return " ".join(parts) or "NOTHING CONFIGURED (all requests refused)"
