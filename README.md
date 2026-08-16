# farm_assistant_hermes

The engine behind **`/farm-assistant-v3`** — an internal pilot that runs the
EU-FarmBook assistant as a Hermes *agent* instead of a single-shot RAG pipeline,
to find out whether a persistent per-user memory makes the assistant materially
better.

It is a sibling of `farm_assistant_um`, not a replacement. `eu-farmbook-frontend`
is only the interface; nothing here is wired into `/farm-assistant` (v2), and
v2's code is untouched.

```
eu-farmbook-frontend  /farm-assistant-v3
        │  /api/farm-assistant-v3/*        (server-side proxy, holds the caller key)
        ▼
   adapter (this repo, :8100)  ──►  django_euf_admin   auth, sessions, transcript, MEMORY
        │                        ──►  scout             retrieval
        ▼
   hermes (:8642, loopback)  ──►  Mistral AI            inference
        └── MCP bridge ──► adapter /internal/tools/*
```

## The two things that make it safe to point at real accounts

**One profile per user, created on first use.** `gateway.multiplex_profiles` is
on and every pilot user is addressed as `/p/<profile>/v1/...`, with their profile
id defaulting to their uuid.

Getting into the pilot is decided by three settings, any one of which admits a
user: **`HERMES_PILOT_EMAIL_DOMAINS`** (anyone with a `@ugent.be` address —
least maintenance), **`HERMES_PILOT_ROSTER_FILE`** (a uuid list re-read every
30s, so `echo <uuid> >> roster.txt` needs no restart), or the static
**`HERMES_PILOT_UUIDS`**. With none configured, nobody is admitted. Their profile directory is written by the adapter on their first turn
(`app/services/provisioning.py`), because Hermes resolves `/p/<profile>/` through
`profiles_to_serve()`, "intentionally lightweight (a directory scan + name
validation only)", on *every request*. Writing the directory is enough, which is
why this needs no docker socket and no shell into the agent container — only the
shared data volume.

Authorization stays manual on purpose: provisioning is automatic, *eligibility*
is not. A uuid off the roster is refused and nothing is created for it. If
provisioning fails, the request is refused rather than falling back to a shared
agent — there is no default-profile fallback and there must never be one, since
Hermes' agent state is per profile home.

**The memory scope is derived, not accepted.** Hermes honours any
`X-Hermes-Session-Key` from a caller holding the API key. The adapter builds that
header itself, from the uuid it got by introspecting the JWT against Django. No
code path lets a client name its own memory scope, and a test asserts the
function has no parameter for one.

## Memory lives in MySQL, not on disk

Hermes' built-in memory writes `MEMORY.md` / `USER.md` into the profile home.
Personal data on an EU platform belongs in the database with the rest of the
account, so that store is **disabled** (`memory:` block in
`hermes-data/config.yaml`) and replaced by:

- **reads** — the adapter loads `about_you`, `custom_instructions` and memory
  notes from `django_euf_admin` and injects them per turn;
- **writes** — the agent calls `remember_about_user`, which POSTs to
  `/chat/user/memory/` with the *caller's* token.

These are the same rows `farm_assistant_um` uses, so a pilot user has one memory
across v2 and v3. No new tables, no migrations, Django untouched.

What survives from the mneme design is the shape, not the storage: a profile the
user writes (`USER.md` → `about_you`), a note file the agent writes and the user
can correct (`MEMORY.md` → memory notes). `/chatbot/api/users/me/memory/documents`
renders the rows in exactly that two-document form.

## Scope is enforced, twice

The assistant answers agriculture, farming, food-systems and EU-FarmBook
questions and declines everything else. That contract is stated in
`hermes-data/SOUL.md` (the agent's constitution) **and** restated by
`app/services/scope.py` in the system message of every turn, after the user's
remembered preferences — so the last thing the model reads is the constraint,
not an attempt to escape it. Prompt-injection carriers are called out explicitly:
retrieved passages, memory, the user's own custom instructions.

There is deliberately **no keyword blocklist**. Off-topic keyword guards were
removed from the Farm Assistant on a locked decision — brittle, and English-only
on a 24-language platform. Enforcement is LLM-first; deterministic code may
accelerate or defer, never reject.

## Multi-hop retrieval: the citation register

The agent decides whether and how often to search, so a turn can contain several
retrievals. That breaks naive citation handling in two ways, both handled in
`tool_server.TurnContext`:

- **Numbering is turn-global.** Each hop continues from the last, and a document
  seen twice keeps its first number. Without this, two hops both start at `[1]`,
  the model cites one and the UI renders the other.
- **The source rail is re-emitted.** `ask.py` tracks the register's version and
  publishes the cumulative list whenever it moves, plus a final sweep after the
  stream ends. A "sent / not sent" latch would show hop 1 while the answer cited
  hop 2.

Each hop also returns a `quality` verdict (`strong` / `weak` / `empty`) computed
with the same helpers and thresholds v2 uses. The difference is what happens
next: **v2 drops weak context, v3 reports it and lets the agent decide** — which
is the entire point of the agent route, and impossible if the model cannot see
how good the results were.

The per-turn loop budget is clamped to `HERMES_MAX_ITERATIONS=6` in
`docker-compose.yml`. Hermes' default is 500, which on a per-token-billed API
with the memory block in every call is a cost hazard, not a feature.

## Capability surface: two tools, no credentials

`search_eu_farmbook` and `remember_about_user`, both reached through an MCP
bridge that forwards to this adapter. No web, terminal, code execution,
sandboxes, cron, kanban or file access. The agent holds no OpenSearch password
and never sees a user JWT.

## Run it

```bash
cp .env.sample .env          # fill CHAT_BACKEND_URL, OPENSEARCH_*, HERMES_PILOT_UUIDS
docker compose up --build -d # or ./run.sh --docker
docker compose exec adapter curl -s localhost:8100/health
```

Profiles appear by themselves as pilot users sign in. `./scripts/seed_profiles.sh`
is optional — it pre-seeds a profile under a readable name, and `--list` shows
what exists.

`./run.sh` alone runs the adapter with reload on `127.0.0.1:8100` against an
already-running agent; `./run.sh --test` runs the suite. To reach the adapter
from the host in a local stack, uncomment the `ports:` block in
`docker-compose.yml`.

## Deployment (nexavion)

Hosted as `hermes.farm-assistant.nexavion.com`, alongside `farm-assistant.nexavion.com`.

```bash
./build_and_push.sh              # ghcr.io/eu-farmbook/farm_assistant_hermes:latest
# on the host: docker compose pull && docker compose up -d
```

Neither container publishes a port. The reverse proxy terminates TLS and
forwards to the **adapter** on 8100; the frontend sets
`FARM_ASSISTANT_V3_API_URL` to that hostname and sends `X-API-Key`.

**The agent's own port stays unpublished and unproxied.** `:8642` is the full
capability surface behind one bearer key — if it is ever routable from outside,
anyone holding that key can talk to any pilot user's agent. There is no reason
to expose it; debug through `docker compose exec`.

Checklist before the first pilot user:

- `REQUIRE_API_KEY=true` and `CHAT_API_KEYS` carries the frontend's key hash
- `CHAT_BACKEND_URL` / `AUTH_BACKEND_URL` point at the **same** Django realm the
  frontend logs into, or every token fails introspection
- at least one of `HERMES_PILOT_EMAIL_DOMAINS` / `HERMES_PILOT_ROSTER_FILE` /
  `HERMES_PILOT_UUIDS` is set — otherwise every request is refused
- the `./hermes-data` volume is mounted into **both** containers — the adapter
  writes profile directories into it
- `OPENSEARCH_*` copied from `farm_assistant_um` so v2 and v3 retrieve identically
- the frontend's `FARM_ASSISTANT_V3_PILOT_UUIDS` matches the profile map

`hermes-data/config.yaml` is a **template** — it carries `__EUF_PROFILE__` and
`__EUF_BRIDGE_KEY__` placeholders that `seed_profiles.sh` substitutes per
profile. The agent does not run from it directly.

## Known limits of the pilot

- **Inference is third-party.** Every turn ships the user's remembered profile to
  Mistral AI. That is why the pilot is internal-only and why `config.yaml`
  documents the way back to self-hosted inference (vLLM needs
  `--enable-auto-tool-choice --tool-call-parser hermes` and `--max-model-len 65536`;
  the platform's current concurrency-tuned config at 16k is below Hermes' 64k floor).
- **Single replica.** Turn context — the caller's token and the parked sources —
  is in-process, keyed by profile. Scaling out means moving it to Valkey.
- **Not promotable as-is.** v3 has no attachments, export, voice, follow-ups or
  platform-stats path, and its scope enforcement has had none of the traffic v2's
  has. It is an experiment, not a candidate.

## Provenance

`app/security.py`, `app/services/auth_service.py`, `app/services/search_service.py`
and `app/services/context_service.py` are **verbatim copies** from
`farm_assistant_um`. Copied, not imported, for the same reason
`agentic_farm_assistant` copied `farm_assistant`: this experiment must not be
able to break the live assistant. If you fix a bug in one, port it deliberately.
