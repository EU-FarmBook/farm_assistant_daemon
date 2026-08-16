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
    HERMES_MULTIPLEX_PROFILES: bool = True
    HERMES_REQUEST_TIMEOUT_SECONDS: float = 180.0

    # --- Pilot roster and profiles -------------------------------------------
    # THE ALLOWLIST: comma-separated user uuids. A uuid absent from it is
    # refused, never served by a shared agent. Membership is deliberately
    # manual — the profile itself is created automatically on first use.
    HERMES_PILOT_UUIDS: str = ""

    # Optional `uuid:profile` naming overrides, for profiles seeded by hand with
    # readable names. Not the roster (a uuid here is implicitly allowed); by
    # default a user's profile id is simply their uuid.
    HERMES_PROFILE_MAP: str = ""

    # The agent's data volume as this container sees it. The adapter writes a
    # profile directory here on a user's first turn; Hermes picks it up on the
    # next request (its profile routing is a live directory scan).
    HERMES_DATA_DIR: str = "/opt/data"

    # --- Memory summary ------------------------------------------------------
    # The settings dialog's "Memory summary / Update" button. This is the ONE
    # place the adapter calls a model directly instead of going through the
    # agent: summarising a user's stored facts must not run an agent loop, must
    # not touch the conversation transcript, and must not be able to trigger a
    # retrieval. A plain completion is the right tool.
    # Blank key = the endpoint returns the cached summary and declines to
    # regenerate, which degrades the button rather than the dialog.
    MISTRAL_API_URL: str = "https://api.mistral.ai"
    MISTRAL_API_KEY: str = ""
    MEMORY_SUMMARY_MODEL: str = "mistral-medium-latest"

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
