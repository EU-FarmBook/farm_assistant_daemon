# app/config.py
#
# Settings for the Hermes adapter.
#
# Deliberately a SUBSET of farm_assistant_um's config: this service owns no
# prompt assembly, no scope routing, no attachments and no export, so the
# settings for those do not exist here. The names that DO appear are spelled
# exactly as they are in farm_assistant_um, because auth_service.py,
# search_service.py and context_service.py are verbatim copies that read them.

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Environment selector ---
    FA_ENV: str = Field("local")  # "local" | "dev" | "prd"

    # --- App ---
    LOG_LEVEL: str = Field("INFO")
    APP_TITLE: str = "Farm Assistant (Hermes) Adapter"
    APP_VERSION: str = "0.1.0"
    ENABLE_DOCS: bool = False

    # --- Django (auth + chat persistence) ---
    # Identical realm to farm_assistant_um: the pilot signs in on the platform,
    # so tokens must be introspected against the SAME Django that minted them.
    CHAT_BACKEND_URL: str = ""
    AUTH_BACKEND_URL: str = ""
    AUTH_TOKEN_INTROSPECTION: bool = True
    VERIFY_SSL: bool = True

    # Chat is a signed-in feature. An unauthenticated caller has no profile to
    # route to (see profile_registry), so anonymous access is not merely
    # discouraged here — it is unrepresentable. Kept as a flag only for local
    # debugging against a stubbed Django.
    REQUIRE_CHAT_AUTH: bool = True

    # --- CORS -----------------------------------------------------------------
    # Comma-separated origin allowlist for a BROWSER client. Empty (the default)
    # installs no CORS middleware at all, which is the right default for a
    # server-to-server API and matches how this service has always behaved —
    # but it means a browser page on any other origin cannot call it, because
    # the preflight goes unanswered. There is deliberately no wildcard: this
    # service is handed platform JWTs, and "any origin may send us a bearer
    # token" is not a thing to switch on by accident.
    CORS_ALLOW_ORIGINS: str = ""

    # --- Caller API-key gate (same scheme as farm_assistant_um) ---
    # csv of `label:sha256hex`. Plaintext keys never touch this config.
    REQUIRE_API_KEY: bool = True
    CHAT_API_KEYS: str = ""
    API_KEY_HEADER: str = "X-API-Key"

    # --- Retrieval (scout / Opensearch_FastAPI_Test) ---
    # The one capability Hermes is given. Same endpoint farm_assistant_um uses,
    # so v3 and v2 retrieve from the same place with the same parameters and any
    # quality difference between them is attributable to the agent, not the index.
    OPENSEARCH_API_URL: str = ""
    OS_RAG_API_PATH: str = "/llm_retrieve"
    OPENSEARCH_API_USR: str = ""
    OPENSEARCH_API_PWD: str = ""
    RETRIEVAL_CANDIDATE_K: int = 10
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 24000

    # Per-item OpenSearch score floor — junk removal, same value as v2.
    RETRIEVAL_MIN_SCORE: float = 1.0
    # Relevance floors, also v2's. NOTE the difference in what they DO: v2 drops
    # contexts below these, v3 reports the verdict to the agent and lets it
    # decide whether to search again. Keep the numbers aligned with
    # farm_assistant_um so "weak" means the same thing in both.
    RELEVANCE_MODE: str = "overlap"          # "overlap" | "semantic"
    RETRIEVAL_DROP_THRESHOLD: float = 0.15   # overlap mode
    SEMANTIC_DROP_THRESHOLD: float = 0.88    # semantic mode; msmarco-specific, re-calibrate per model

    # Public platform URL used to build citation links back to knowledge objects.
    PLATFORM_PUBLIC_URL: str = "https://eufarmbook.eu"

    # --- Hermes ---
    # The agent's OpenAI-compatible API. Loopback in local runs; the compose
    # network name in containers. NEVER expose this port publicly: it is the
    # agent's entire capability surface behind one bearer key.
    HERMES_API_URL: str = "http://127.0.0.1:8642"
    HERMES_API_KEY: str = ""
    # Per-user isolation. With multiplex_profiles on, each pilot user's agent is
    # addressed as /p/<profile>/v1/... and owns its own MEMORY.md and USER.md.
    # Off = every request lands on the default profile and therefore shares one
    # memory file, which defeats the point of the pilot; see README.
    # The agent's model. Substituted into each profile's config.yaml, so this is
    # the one place to change when comparing models. Must be a model the
    # configured provider serves, with reliable TOOL CALLING and at least
    # Hermes' 64k context floor — the agent is useless without the first and
    # stalls on the second. No default: see MEMORY_SUMMARY_MODEL below.
    HERMES_MODEL: str = ""
    HERMES_MULTIPLEX_PROFILES: bool = True
    HERMES_REQUEST_TIMEOUT_SECONDS: float = 180.0

    # --- Access ---------------------------------------------------------------
    # OPEN ACCESS: any authenticated EU-FarmBook user may chat, and their agent
    # is created on their first message. No roster, no operator step.
    #
    # What this turns off is a *bound*, not a login check: identity is still
    # verified against Django on every request. What becomes unbounded is spend
    # (every turn is billed to LLM_API_KEY), disk (one profile directory per
    # user who ever visits), and how many people's remembered profiles are sent
    # to a third party. RATE_LIMIT_* below is what keeps the first of those
    # survivable — do not run open access with the limiter disabled.
    HERMES_OPEN_ACCESS: bool = False

    # --- Per-user rate limit --------------------------------------------------
    # Counted per verified uuid, in-process (this service is single-replica by
    # design — see tool_server). An agent turn is several model calls, so these
    # are deliberately lower than a chat-completion service would use.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_TURNS_PER_MIN: int = 6
    RATE_LIMIT_TURNS_PER_DAY: int = 120

    # Hard ceiling on how many agents can ever exist, as a disk and blast-radius
    # bound under open access. 0 = unlimited. When reached, existing users keep
    # working and new ones are refused with a clear log line.
    MAX_PROFILES: int = 0

    # --- Roster (optional; ignored when HERMES_OPEN_ACCESS is true) ------------
    # THREE ways to admit a user when access is NOT open. Configure at least one
    # or nobody gets in — the gate fails closed. The profile itself is always
    # created automatically on first use.
    #
    # 1. Email domains, comma-separated (`ugent.be,nexavion.com`). Least
    #    maintenance: nobody's uuid has to be looked up. Only applied when the
    #    verified token actually carries an email claim — check the startup log.
    HERMES_PILOT_EMAIL_DOMAINS: str = ""

    # 2. A file of uuids, one per line (`#` comments allowed), re-read every 30s.
    #    Adding someone is `echo <uuid> >> roster.txt` — no restart, no redeploy.
    #    Put it on the mounted volume, e.g. /opt/data/pilot-roster.txt.
    HERMES_PILOT_ROSTER_FILE: str = ""

    # 3. Static list in this file. Requires a restart to change.
    HERMES_PILOT_UUIDS: str = ""

    # Optional `uuid:profile` naming overrides, for profiles seeded by hand with
    # readable names. Not the roster (a uuid here is implicitly allowed); by
    # default a user's profile id is simply their uuid.
    HERMES_PROFILE_MAP: str = ""

    # The agent's data volume as this container sees it. The adapter writes a
    # profile directory here on a user's first turn; Hermes picks it up on the
    # next request (its profile routing is a live directory scan).
    HERMES_DATA_DIR: str = "/opt/data"

    # Address the user by first name when the verified token carries one. Set
    # false to send the inference provider nothing that names the person: the
    # remembered profile alone is pseudonymous, a name is not.
    INCLUDE_USER_NAME: bool = True

    # --- Attachments (documents only) -----------------------------------------
    # Extracted at upload and held in-process, so these bound memory as much as
    # they bound the upload. Images are not supported in v3 — that needs the
    # vision model wired, which is separate work.
    ATTACHMENT_MAX_BYTES: int = 15 * 1024 * 1024
    ATTACHMENT_MAX_CHARS: int = 120_000

    # --- Inference provider ---------------------------------------------------
    # ONE OpenAI-compatible endpoint, used two ways: the agent reaches it through
    # the `providers` block in hermes-data/config.yaml (which is handed this key
    # via each profile's .env), and the adapter calls it directly for the four
    # side-features that must NOT run an agent turn — the memory summary, the
    # memory-write guard, the opening suggestions and the follow-up chips. None
    # of those may run a tool loop, enter the transcript, or trigger a retrieval,
    # which is why they are plain completions rather than agent calls.
    #
    # Currently Scaleway Generative APIs: EU-hosted, which matters because every
    # turn ships the user's remembered profile to whoever serves inference.
    # Swapping provider is these two values plus the model ids; nothing else.
    #
    # No trailing /v1 here — the call sites append `/v1/chat/completions`. The
    # agent's own base_url in config.yaml DOES include it. A project-scoped
    # Scaleway endpoint (https://api.scaleway.ai/<project-id>) also works and is
    # what the console hands out; both forms answer.
    LLM_API_URL: str = "https://api.scaleway.ai"
    # Blank key = the memory summary returns its cached value, the memory guard
    # refuses every write (it fails closed), and suggestions fall back to the
    # static set. Degraded, not broken.
    LLM_API_KEY: str = ""
    # The cheap model for those four side-features. Deliberately no default: a
    # wrong model id fails at call time with a provider error that reads like a
    # bug in this service. Set it from what the key can actually reach —
    #   curl -s $LLM_API_URL/v1/models -H "Authorization: Bearer $LLM_API_KEY"
    #
    # It MUST NOT be a reasoning model. These calls ask for a single word at
    # max_tokens=5, and a reasoning model spends that budget on its `reasoning`
    # field and returns content: "" — which the memory guard reads as "cannot
    # validate" and fails closed on, silently refusing every memory write.
    # Measured on Scaleway: of nine chat models, only llama-3.3-70b-instruct
    # answered directly; gpt-oss-120b, glm-5.2, gemma-4, deepseek-v4-flash and
    # both qwen3.x models all came back empty.
    MEMORY_SUMMARY_MODEL: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    def model_post_init(self, __context) -> None:
        # Resolve a blank backend URL from FA_ENV, exactly as farm_assistant_um
        # does. This is not a convenience: with no backend URL, auth_service
        # falls back to an UNVERIFIED JWT decode, so a blank value on a public
        # host would let anyone forge a token carrying a rostered uuid and reach
        # that user's agent and memory. See also the startup check in main.py,
        # which refuses to run non-local without introspection.
        backend_by_env = {
            "local": "http://127.0.0.1:8000",
            "dev": "https://backend-admin.dev.farmbook.ugent.be",
            "prd": "https://backend-admin.prd.farmbook.ugent.be",
        }
        if not self.CHAT_BACKEND_URL:
            env = (self.FA_ENV or "local").lower()
            self.CHAT_BACKEND_URL = backend_by_env.get(env, backend_by_env["local"])

        for attr in ("CHAT_BACKEND_URL", "AUTH_BACKEND_URL", "OPENSEARCH_API_URL",
                     "HERMES_API_URL", "PLATFORM_PUBLIC_URL"):
            value = getattr(self, attr, "") or ""
            if value:
                setattr(self, attr, value.rstrip("/"))

    def auth_is_verified(self) -> bool:
        """True when tokens are actually introspected rather than trusted."""
        return bool(self.AUTH_TOKEN_INTROSPECTION and (self.AUTH_BACKEND_URL or self.CHAT_BACKEND_URL))

    def cors_origins(self) -> list[str]:
        """Parse CORS_ALLOW_ORIGINS into a list. Wildcards are rejected, not honoured."""
        out = []
        for raw in (self.CORS_ALLOW_ORIGINS or "").split(","):
            origin = raw.strip().rstrip("/")
            if not origin or origin == "*":
                continue
            out.append(origin)
        return out

    def api_keys_map(self) -> dict[str, str]:
        """
        Parse CHAT_API_KEYS (csv of `label:sha256hex`) into {sha256hex: label}.
        Malformed entries are skipped rather than raising: one bad pair should
        not take the service down at import time.
        """
        out: dict[str, str] = {}
        for raw in (self.CHAT_API_KEYS or "").split(","):
            entry = raw.strip()
            if not entry or ":" not in entry:
                continue
            label, _, digest = entry.partition(":")
            label, digest = label.strip(), digest.strip().lower()
            if label and digest:
                out[digest] = label
        return out

    def profile_map(self) -> dict[str, str]:
        """Parse HERMES_PROFILE_MAP (csv of `uuid:profile`) into {uuid: profile}."""
        out: dict[str, str] = {}
        for raw in (self.HERMES_PROFILE_MAP or "").split(","):
            entry = raw.strip()
            if not entry or ":" not in entry:
                continue
            uuid, _, profile = entry.partition(":")
            uuid, profile = uuid.strip(), profile.strip()
            if uuid and profile:
                out[uuid] = profile
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
