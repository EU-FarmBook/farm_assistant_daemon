# app/main.py

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers.ask import router as ask_router
from app.routers.files import router as files_router
from app.routers.follow_ups import router as follow_ups_router
from app.routers.memory import router as memory_router
from app.routers.sessions import router as sessions_router
from app.routers.tools import router as tools_router
from app.security import path_requires_key, resolve_api_key_label
from app.services.hermes_client import health as hermes_health
from app.services.profile_registry import gate_description, pilot_size
from app.services.provisioning import (
    profile_count,
    refresh_all_profiles,
    write_bridge_key,
    write_default_config,
)

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
    write_bridge_key()
    # The default profile's config is generated from the template, so an edit to
    # the template reaches it too. See provisioning.write_default_config for why
    # the default profile is worth hardening at all.
    write_default_config()
    refreshed = refresh_all_profiles()
    logger.info(
        "Profiles: %d total, %d refreshed from the current template",
        profile_count(), refreshed,
    )
    logger.info("Pilot gate: %s", gate_description())
    logger.info("Auth realm: %s", S.AUTH_BACKEND_URL or S.CHAT_BACKEND_URL)
    if not S.LLM_API_KEY:
        logger.info("LLM_API_KEY unset — the memory summary will show its cached value only.")
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

    /internal/tools/* is not gated here — it carries its own bridge key. It is
    kept off the public router by the `!PathPrefix(`/internal/`)` clause in the
    Traefik rule in docker-compose.yml; that clause is the enforcement, and this
    comment used to assert the isolation while the shipped rule was host-only
    and published the tool surface to the internet.
    """
    if S.REQUIRE_API_KEY and path_requires_key(request.url.path, request.method):
        presented = request.headers.get(S.API_KEY_HEADER, "")
        label = resolve_api_key_label(presented, S.api_keys_map())
        if not label:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key."})
        request.state.api_key_label = label

    return await call_next(request)


# Added AFTER api_key_middleware, which makes it the OUTERMOST layer: a
# preflight is answered before the key gate ever sees it. That ordering matters
# because a browser cannot attach X-API-Key to an OPTIONS request — the gate
# exempts OPTIONS for exactly that reason, but exempting a request is not the
# same as answering it, and nothing answered it before.
#
# Off unless an allowlist is configured, so the default stays a server-to-server
# API. No wildcard: this service receives platform JWTs.
_cors_origins = S.cors_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        # Bearer tokens travel in a header, not a cookie, so credentialed
        # requests are not needed — and allowing them would forbid ever
        # relaxing allow_origins.
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", S.API_KEY_HEADER],
        # So a browser client can read Retry-After on a 429.
        expose_headers=["Retry-After", "Content-Disposition"],
        max_age=600,
    )
    logger.info("CORS enabled for: %s", ", ".join(_cors_origins))


@app.get("/health", include_in_schema=False)
async def health():
    """
    Liveness plus a readable summary. Always 200 while the process is up.

    Deliberately not a dependency check: killing a healthy process because
    something downstream is unreachable turns one outage into two. Use
    /health/ready for the question "can this serve a turn right now".
    """
    agent_up = await hermes_health()
    return {
        "status": "ok" if agent_up else "degraded",
        "agent": "up" if agent_up else "down",
        "pilot_profiles": pilot_size(),
        "version": S.APP_VERSION,
    }


@app.get("/health/ready", include_in_schema=False)
async def readiness():
    """
    Readiness: 200 only if a turn could actually be served, 503 otherwise.

    /health answers 200 whether or not the agent is reachable, so the compose
    healthcheck reported green while every turn failed. That is right for
    liveness and useless for a load balancer, so the two are separated and the
    healthcheck points here.

    Checked: the agent responds, and the two settings with no default are set —
    an unset HERMES_MODEL makes provisioning refuse every new user, and an unset
    MEMORY_SUMMARY_MODEL silently disables the memory guard, suggestions,
    follow-ups and the memory summary.
    """
    agent_up = await hermes_health()
    checks = {
        "agent": agent_up,
        "agent_model_set": bool(S.HERMES_MODEL.strip()),
        "utility_model_set": bool(S.MEMORY_SUMMARY_MODEL.strip()),
        "provider_key_set": bool(S.LLM_API_KEY),
        "retrieval_configured": bool(S.OPENSEARCH_API_URL and S.OPENSEARCH_API_PWD),
    }
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


app.include_router(ask_router)
app.include_router(sessions_router)
app.include_router(memory_router)
app.include_router(files_router)
app.include_router(follow_ups_router)
app.include_router(tools_router)
