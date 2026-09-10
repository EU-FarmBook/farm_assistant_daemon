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
   hermes (:8642, loopback)  ──►  Scaleway (EU)         inference
        └── MCP bridge ──► adapter /internal/tools/*
```

## The two things that make it safe to point at real accounts

**One profile per user, created on first use.** `gateway.multiplex_profiles` is
on and every pilot user is addressed as `/p/<profile>/v1/...`, with their profile
id defaulting to their uuid.

With **`HERMES_OPEN_ACCESS=true`** there is no enrolment at all: a user arrives,
starts chatting, and their agent is created on the first message. Identity is
still verified against Django on every request — open access removes the roster,
not the login, and an unauthenticated caller has no agent to be routed to.

What replaces the roster as a bound is **`RATE_LIMIT_*`** (turns per user per
minute and per day, refused before any model call) and optionally
**`MAX_PROFILES`**. Do not run open access with the limiter off: a turn is an
agent loop of several billed calls, so one account with a valid token can spend
without limit, and the first sign is the invoice.

For a closed pilot instead, set `HERMES_OPEN_ACCESS=false` and use any of
`HERMES_PILOT_EMAIL_DOMAINS`, `HERMES_PILOT_ROSTER_FILE` (re-read every 30s, so
`echo <uuid> >> roster.txt` needs no restart) or `HERMES_PILOT_UUIDS`. With none
configured, nobody is admitted — the roster path fails closed. Their profile directory is written by the adapter on their first turn
(`app/services/provisioning.py`), because Hermes resolves `/p/<profile>/` through
`profiles_to_serve()`, "intentionally lightweight (a directory scan + name
validation only)", on *every request*. Writing the directory is enough, which is
why this needs no docker socket and no shell into the agent container — only the
shared data volume.

**A provisioned profile needs three files, not two.** `config.yaml`, `SOUL.md`,
and its own `.env`. Under multiplexing Hermes resolves a named profile's
credentials inside that profile's secret scope and refuses to borrow the
listener's — `_expected_api_key()`: *"Named profiles must fail closed rather than
inherit the listener owner's key."* Without `<profile>/.env` carrying
`API_SERVER_KEY` (and the provider key, scoped the same way), every request
returns:

```
API server rejected request for profile '<id>': no profile-scoped
API_SERVER_KEY is configured
```

The adapter writes and repairs all three, so rotating `HERMES_API_KEY` fixes
itself on the next turn.

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

## Retrieve first, then let the agent search again

v2 retrieves before it generates, so it cannot answer ungrounded. A pure agent
decides for itself — and sometimes does not look at all, which produced an answer
asserting EU-FarmBook had no material on pig manure without a single search
having run. v3 therefore keeps v2's floor and adds to it:

1. Every substantive turn (anything over `_PREFETCH_MIN_CHARS`) is retrieved for
   before the model runs, through **the same `search_eu_farmbook` the agent
   calls** — so pre-retrieved and agent-retrieved passages share one citation
   register and one numbering.
2. The passages arrive numbered, with their **quality verdict**. Where v2
   silently drops a weak set and answers anyway, v3 says it was weak and asks the
   agent to re-query with tighter terms.
3. The agent can search again for a second part of a question, or a better angle.
   Later hops continue the numbering, so `[1]` never changes meaning mid-answer.
4. A short follow-up ("and for maize?") is retrieved with the previous user turn
   prepended — a bare anaphor retrieves nothing useful. v2 spends an LLM call
   resolving this; here a heuristic is enough because the agent can re-query when
   it guesses wrong.

Greetings skip the search on length alone, never a keyword list — those are
brittle and English-only, which is why v2's were removed.

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
sandboxes, cron or file access. The agent holds no OpenSearch password and never
sees a user JWT.

Verify rather than assume, with Hermes' own resolver:

```bash
docker run --rm -v "$PWD/hermes-data/config.yaml:/tmp/cfg.yaml:ro" \
  --entrypoint python nousresearch/hermes-agent:latest -c "
import sys, yaml; sys.path.insert(0, '/opt/hermes')
from hermes_cli.tools_config import _get_platform_tools
print(sorted(_get_platform_tools(yaml.safe_load(open('/tmp/cfg.yaml')), 'api_server')))"
# -> ['eu-farmbook']
```

`platform_toolsets.api_server: []` is honoured as written: `_get_platform_tools`
only falls back to platform defaults when the key is `None` or not a list, and an
empty list is a list. If it ever fell back, the agent would silently gain the
full default toolset — terminal included — so re-run the check after touching
that block.

**Expect this warning in the agent's logs, and ignore it:**

> `API server is network-accessible (0.0.0.0) AND the terminal backend is
> 'local' (unsandboxed).`

It fires on the combination of bind address and terminal backend without
checking whether the terminal toolset is enabled for the platform on that port.
Ours is not (see above). `0.0.0.0` is required for the adapter to reach the agent
across the compose network; the port stays unpublished and off `traefik-net`.

## Driving it from another application

Two doors onto the same turn — same gates, same retrieval, same citation register,
same memory:

```bash
# Streaming, when a person is watching it arrive:
GET  /chatbot/api/chats/{session_id}/message/stream?q=...
     -> status, sources, grounding, token*, final, timing, done

# One request, one answer, when nobody is:
POST /chatbot/api/chats/message   {"q": "...", "session_id": "..."}
     -> {"ok", "answer", "sources", "grounding", "timing_ms"}
```

Both take `Authorization: Bearer <platform JWT>` and, when `REQUIRE_API_KEY=true`,
`X-API-Key`. A turn measures 8-50s, so allow at least a 120s client timeout on the
POST door; it has no progress signal, and a dropped connection loses the whole
answer rather than part of it. The stream's exact wire contract — `token` is a bare
string, `sources` replaces rather than appends, `data:` lines need spec reassembly —
is published in the OpenAPI `description` for that endpoint.

Browser clients need `CORS_ALLOW_ORIGINS` set; with it empty this is a
server-to-server API and a preflight goes unanswered.

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

Neither container publishes a port. Traefik terminates TLS and reaches the
**adapter** over the shared external `traefik-net`, on 8100 — same label pattern
as `farm_assistant` and its arena siblings on this host. The frontend sets
`FARM_ASSISTANT_V3_API_URL` to that hostname and sends `X-API-Key`.

**The agent is on the internal network only**, with `traefik.enable=false` and no
membership of `traefik-net`. `:8642` is its full capability surface behind one
bearer key — anyone reaching it could talk to any pilot user's agent. The adapter
is deliberately the only container on both networks: it is the boundary. Debug
the agent through `docker compose exec`, never by exposing it.

Checklist before the first pilot user — `.env.sample` carries these as its
defaults, so the short version is "copy it and fill the blanks":

- **two** env files, not one: `.env` (adapter) and `hermes-data/.env` (agent).
  `HERMES_API_KEY` must equal `API_SERVER_KEY`, and `LLM_API_KEY` must be in
  both — the agent resolves credentials in its own scope and cannot borrow the
  adapter's
- `FA_ENV=prd` (or `dev`) — **not** `local`. Any non-local value makes the
  service refuse to start unless introspection is configured, which is the
  point: a blank auth realm means tokens are decoded without verification
- `REQUIRE_API_KEY=true` and `CHAT_API_KEYS` carries the frontend's key hash
- `CHAT_BACKEND_URL` / `AUTH_BACKEND_URL` point at the **same** Django realm the
  frontend logs into, or every token fails introspection. Blank resolves them
  from `FA_ENV`
- `HERMES_OPEN_ACCESS=true` (or, for a closed pilot, one of the roster settings
  — with none of them nobody is admitted)
- `RATE_LIMIT_ENABLED=true` — the only thing bounding spend under open access —
  and a real `MAX_PROFILES` ceiling if the host is public
- `HERMES_MODEL` and `MEMORY_SUMMARY_MODEL` are both set, and the second is
  **not** a reasoning model (it is called with `max_tokens=5`; a reasoning model
  returns empty content and the memory guard then refuses every write)
- the `./hermes-data` volume is mounted into **both** containers — the adapter
  writes profile directories into it
- `OPENSEARCH_*` copied from `farm_assistant_um` so v2 and v3 retrieve identically

`hermes-data/config.template.yaml` is the **template** — it carries
`__EUF_PROFILE__` and `__EUF_MODEL__` placeholders — deliberately not the bridge
key, which the MCP server reads per call from `bridge.key` (0600) so that no
rendered config carries a credential — and the
adapter substitutes them per profile on that user's first turn.

Its sibling `hermes-data/config.yaml` is **generated** from it and belongs to the
agent: that path is the *default* profile's live config, and the agent rewrites
it in place whenever the on-disk `_config_version` is behind the image's,
stripping the comments. Editing it is pointless — edit the template. It stays
tracked because the agent boots before the adapter, and a missing one on a fresh
volume would let the agent create an unhardened default profile (built-in memory
on, full platform toolset) and serve it until the next restart.

## Known limits of the pilot

- **Inference is third-party.** Every turn ships the user's remembered profile to
  Scaleway Generative APIs. EU-hosted, which is why it is preferred over a US
  provider here, but still third-party — which is why the pilot is internal-only
  and why `config.yaml`
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
