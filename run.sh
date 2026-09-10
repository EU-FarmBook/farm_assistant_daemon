#!/usr/bin/env bash
#
# Local runner. Mirrors farm_assistant_um's run.sh shape so muscle memory carries
# over, minus the backend switches this service does not have.
#
#   ./run.sh              adapter with reload on :8100 (expects hermes already up)
#   ./run.sh --docker     both containers via docker compose
#   ./run.sh --test       pytest
#   ./run.sh --stop       stop the compose stack

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:-}" in
  --docker)
    docker compose up --build -d
    docker compose ps
    echo
    echo "Adapter  http://127.0.0.1:8100/health"
    echo "Agent    http://127.0.0.1:8642/health  (loopback only)"
    echo "Next:    ./scripts/seed_profiles.sh"
    ;;
  --stop)
    docker compose down
    ;;
  --test)
    # Needs requirements-dev.txt (pytest is not in the runtime manifest).
    [[ -d .venv ]] && source .venv/bin/activate
    pytest -q
    ;;
  *)
    [[ -d .venv ]] && source .venv/bin/activate
    # /opt/data is the CONTAINER's view of the volume and does not exist on the
    # host, so a bare run has to be pointed at the checkout or provisioning
    # writes nowhere. An explicit HERMES_DATA_DIR still wins.
    export HERMES_DATA_DIR="${HERMES_DATA_DIR:-$PWD/hermes-data}"
    exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8100
    ;;
esac
