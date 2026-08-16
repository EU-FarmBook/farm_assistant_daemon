# app/services/profile_registry.py
"""
uuid -> Hermes profile, and the pilot allowlist.

Two separate questions, deliberately answered by different mechanisms:

**Is this user allowed?** `HERMES_PILOT_UUIDS` — an explicit roster. A uuid that
is not on it is refused. There is no "default profile" fallback and there must
never be one: Hermes' agent state is per profile home, so two users sharing a
profile share their sessions and anything the agent writes.

**Does their agent exist yet?** Not a question the operator should have to
answer. The profile is created on first use by `provisioning.ensure_profile()`,
because the gateway discovers profiles by directory scan on every request. See
that module for why this needs no docker socket.

`HERMES_PROFILE_MAP` is still honoured for profiles seeded by hand with readable
names (`uuid:alice`); it is now a naming override, not the roster.
"""

import logging
import re
from typing import Optional

from app.config import get_settings
from app.services import provisioning

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.profiles")

# Identical to Hermes' own `_PROFILE_ID_RE` (hermes_cli/profiles.py). A profile
# id becomes a URL path segment and a directory name, so anything outside this
# charset must never reach either.
_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ProfileNotProvisioned(Exception):
    """The caller is authenticated but not part of the pilot, or provisioning failed."""

    def __init__(self, user_uuid: str, reason: str = "not on the pilot roster"):
        self.user_uuid = user_uuid
        self.reason = reason
        super().__init__(f"No Hermes profile for user {user_uuid}: {reason}")


def is_valid_profile_name(name: str) -> bool:
    return bool(name) and bool(_PROFILE_RE.match(name))


def _roster() -> set[str]:
    settings = get_settings()
    listed = {
        entry.strip()
        for entry in (settings.HERMES_PILOT_UUIDS or "").split(",")
        if entry.strip()
    }
    # A uuid named in the naming-override map is implicitly on the roster —
    # otherwise hand-seeded profiles would be unreachable.
    return listed | set(settings.profile_map())


def is_pilot_user(user_uuid: Optional[str]) -> bool:
    return bool(user_uuid) and user_uuid in _roster()


def resolve_profile(user_uuid: Optional[str], *, provision: bool = True) -> str:
    """
    Return the Hermes profile for a VERIFIED user uuid, creating it if needed.

    The uuid must come from auth_service.resolve_user_uuid() — never from a
    request header or body. The whole isolation guarantee reduces to that:
    whoever controls this string controls which agent is addressed.
    """
    if not user_uuid:
        raise ProfileNotProvisioned("<anonymous>", "not authenticated")

    if not is_pilot_user(user_uuid):
        # The normal case for everyone outside the pilot, so info, not error.
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
    return len(_roster())
