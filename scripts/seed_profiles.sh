#!/usr/bin/env bash
#
# OPTIONAL operator tool. You do not need to run this to add a pilot user.
#
# The adapter provisions a profile automatically on a user's first turn (see
# app/services/provisioning.py): Hermes resolves /p/<profile>/ with a live
# directory scan, so writing the directory is enough — no CLI, no restart.
# Adding someone to the pilot needs no edit at all under HERMES_OPEN_ACCESS.
#
# This script remains useful for two things:
#   * pre-seeding a profile with a readable name (HERMES_PROFILE_MAP), e.g. for
#     a demo account, so it exists before anyone signs in;
#   * `--list`, to see what is provisioned right now.
#
# Reads HERMES_PROFILE_MAP from ../.env (csv of `uuid:profile`), creates each
# profile inside the running agent container, and seeds it with:
#   - SOUL.md            the shared scope contract (same file for everyone)
#   - config.yaml        rendered from config.template.yaml, with this profile's
#                        own MCP env and model. Carries NO credential: the
#                        bridge key is read at call time from bridge.key (0600)
#   - .env               its profile-scoped API_SERVER_KEY and provider key
#
# It does NOT seed memory. Memory lives in MySQL under django_euf_admin; the
# agent's own memory store is disabled (see hermes-data/config.yaml).
#
# Usage:
#   ./scripts/seed_profiles.sh            # create/refresh every mapped profile
#   ./scripts/seed_profiles.sh --list     # show what is provisioned now

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${HERMES_CONTAINER:-euf-hermes}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "error: container '$CONTAINER' is not running. Start it with: docker compose up -d hermes" >&2
  exit 1
fi

if [[ "${1:-}" == "--list" ]]; then
  docker exec "$CONTAINER" hermes profile list
  exit 0
fi

# shellcheck disable=SC1091
PROFILE_MAP="$(grep -E '^HERMES_PROFILE_MAP=' "$HERE/.env" | cut -d= -f2- || true)"
BRIDGE_KEY="$(grep -E '^API_SERVER_KEY=' "$HERE/hermes-data/.env" | cut -d= -f2- || true)"
MODEL="$(grep -E '^HERMES_MODEL=' "$HERE/.env" | cut -d= -f2- || true)"
LLM_KEY="$(grep -E '^LLM_API_KEY=' "$HERE/.env" | cut -d= -f2- || true)"

if [[ -z "$PROFILE_MAP" ]]; then
  echo "Nothing to pre-seed: HERMES_PROFILE_MAP is empty in $HERE/.env." >&2
  echo "That is the normal case — profiles listed in HERMES_PILOT_UUIDS are" >&2
  echo "created automatically on each user's first turn. Use this script only" >&2
  echo "to pre-seed a named profile, or run '--list' to inspect." >&2
  exit 0
fi

if [[ -z "$BRIDGE_KEY" ]]; then
  echo "error: API_SERVER_KEY missing from $HERE/hermes-data/.env." >&2
  exit 1
fi

if [[ -z "$MODEL" ]]; then
  echo "error: HERMES_MODEL is empty in $HERE/.env — the template's __EUF_MODEL__" >&2
  echo "       placeholder would be left unsubstituted." >&2
  exit 1
fi

IFS=',' read -ra PAIRS <<< "$PROFILE_MAP"
for pair in "${PAIRS[@]}"; do
  uuid="${pair%%:*}"
  profile="${pair##*:}"
  uuid="$(echo "$uuid" | xargs)"
  profile="$(echo "$profile" | xargs)"

  [[ -z "$uuid" || -z "$profile" ]] && continue

  if ! [[ "$profile" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]]; then
    echo "skip: '$profile' is not a valid profile name (a-z 0-9 _ -)" >&2
    continue
  fi

  echo "==> $profile   (uuid ${uuid:0:8}…)"

  if docker exec "$CONTAINER" hermes profile list 2>/dev/null | grep -qw "$profile"; then
    echo "    exists, refreshing config"
  else
    docker exec "$CONTAINER" hermes profile create "$profile"
  fi

  # Each profile gets the template with its OWN EUF_PROFILE substituted in.
  # That value is how the adapter knows which in-flight turn a tool call belongs
  # to, so it must never be copied unchanged between profiles.
  #
  # Read from config.template.yaml, NOT config.yaml: the latter is the default
  # profile's live config, which the agent rewrites in place on startup.
  #
  # Three files, not two — a profile also needs its own .env, because Hermes
  # resolves a named profile's credentials in that profile's secret scope and
  # refuses to borrow the listener's ("Named profiles must fail closed rather
  # than inherit the listener owner's key").
  #
  # This duplicates app/services/provisioning.py, which is authoritative and
  # runs on every first turn. Keep the two in step, or delete this branch and
  # let a first message do the work.
  docker exec \
    -e SEED_PROFILE="$profile" \
    -e SEED_BRIDGE_KEY="$BRIDGE_KEY" \
    -e SEED_MODEL="$MODEL" \
    -e SEED_LLM_KEY="$LLM_KEY" \
    "$CONTAINER" sh -c '
      set -e
      home=/opt/data/profiles/"$SEED_PROFILE"
      mkdir -p "$home"
      cp /opt/data/SOUL.md "$home/SOUL.md"
      sed -e "s|__EUF_PROFILE__|$SEED_PROFILE|g" \
          -e "s|__EUF_MODEL__|$SEED_MODEL|g" \
          /opt/data/config.template.yaml > "$home/config.yaml"
      # if/then, not `grep && {...}`: under `set -e` a non-matching grep in an
      # && list would abort the script on the SUCCESS path.
      if grep -q "__EUF_" "$home/config.yaml"; then
        echo "    ERROR: placeholders left unsubstituted" >&2
        exit 1
      fi
      umask 077
      {
        echo "# Written by seed_profiles.sh. The adapter regenerates this on the"
        echo "# next turn if the keys change."
        echo "API_SERVER_KEY=$SEED_BRIDGE_KEY"
        [ -n "$SEED_LLM_KEY" ] && echo "LLM_API_KEY=$SEED_LLM_KEY"
      } > "$home/.env"
    '
  echo "    seeded SOUL.md + config.yaml + .env"
done

echo
echo "Done. Verify with: ./scripts/seed_profiles.sh --list"
