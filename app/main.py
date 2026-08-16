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
from app.services.profile_registry import pilot_size

S = get_settings()
logging.basicConfig(level=getattr(logging, S.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("farm-assistant-hermes")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # A pilot with no profile map answers nobody; better to see it in the logs at
    # boot than to debug a wall of 403s.
    if not pilot_size():
        logger.warning("HERMES_PROFILE_MAP is empty — every chat request will be refused.")
    if not S.CHAT_BACKEND_URL:
        logger.warning("CHAT_BACKEND_URL is unset — auth introspection and memory are disabled.")
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
