# app/main.py

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers.ask import router as ask_router
from app.routers.memory import router as memory_router
from app.routers.sessions import router as sessions_router
from app.routers.tools import router as tools_router
from app.security import path_requires_key, resolve_api_key_label
from app.services.hermes_client import health as hermes_health
from app.services.profile_registry import gate_description, pilot_size
from app.services.provisioning import profile_count, refresh_all_profiles

S = get_settings()
logging.basicConfig(level=getattr(logging, S.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("farm-assistant-hermes")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail closed rather than serve unverified identities.
    #
    # With introspection off or no auth backend, auth_service falls back to
    # decoding the JWT WITHOUT verifying it — fine for bare local dev against a
    # stubbed Django, catastrophic on a public host: the uuid is the only thing
    # deciding which agent and whose memory a request reaches, so an unverified
    # one lets anyone forge their way into a pilot user's profile. Refuse to
    # start instead of running in that state.
    if (S.FA_ENV or "local").lower() != "local" and not S.auth_is_verified():
        raise RuntimeError(
            "Refusing to start: FA_ENV=%s but token introspection is not configured "
            "(AUTH_TOKEN_INTROSPECTION=%s, backend=%r). Tokens would be trusted "
            "without verification." % (S.FA_ENV, S.AUTH_TOKEN_INTROSPECTION,
                                       S.AUTH_BACKEND_URL or S.CHAT_BACKEND_URL)
        )

    # A pilot with an empty roster answers nobody; better to see it in the logs
    # at boot than to debug a wall of 403s.
    # Reconcile existing profiles with the current template before serving, so
    # a deploy takes effect at once rather than per-user on first message.
    refreshed = refresh_all_profiles()
    logger.info(
        "Profiles: %d total, %d refreshed from the current template",
        profile_count(), refreshed,
    )
    logger.info("Pilot gate: %s", gate_description())
    logger.info("Auth realm: %s", S.AUTH_BACKEND_URL or S.CHAT_BACKEND_URL)
    if not S.MISTRAL_API_KEY:
        logger.info("MISTRAL_API_KEY unset — the memory summary will show its cached value only.")
    yield


app = FastAPI(
    title=S.APP_TITLE,
    version=S.APP_VERSION,
    docs_url="/docs" if S.ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if S.ENABLE_DOCS else None,
    lifespan=lifespan,
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Caller-key gate, same scheme and same module as farm_assistant_um: a valid
    login is who the user is, the API key is whether this caller is sanctioned.

    /internal/tools/* is not gated here — it carries its own bridge key and is
    never routed from outside the compose network.
    """
    if S.REQUIRE_API_KEY and path_requires_key(request.url.path, request.method):
        presented = request.headers.get(S.API_KEY_HEADER, "")
        label = resolve_api_key_label(presented, S.api_keys_map())
        if not label:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key."})
        request.state.api_key_label = label

    return await call_next(request)


@app.get("/health", include_in_schema=False)
async def health():
    agent_up = await hermes_health()
    return {
        "status": "ok" if agent_up else "degraded",
        "agent": "up" if agent_up else "down",
        "pilot_profiles": pilot_size(),
        "version": S.APP_VERSION,
    }


app.include_router(ask_router)
app.include_router(sessions_router)
app.include_router(memory_router)
app.include_router(tools_router)
