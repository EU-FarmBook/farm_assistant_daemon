# app/routers/sessions.py
"""
Chat sessions and transcript, passed through to django_euf_admin unchanged.

v3 stores nothing of its own. Sessions, titles and the message transcript live
where v2 puts them, with the caller's own token, so a pilot user's chat history
is one history — not a second one that only exists inside the experiment.
"""

import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.schemas import ChatTurnLogIn
from app.services.auth_service import resolve_user_uuid

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.sessions")
router = APIRouter(prefix="/chatbot/api/chats", tags=["Chats"])

_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0)


async def _require_token(request: Request) -> str:
    auth_token = request.headers.get("Authorization", "")
    user_uuid = await resolve_user_uuid(auth_token) if auth_token else None
    if not user_uuid:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return auth_token


async def _django(method: str, path: str, auth_token: str, **kwargs) -> Dict[str, Any]:
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend is not configured.")
    headers = {"Authorization": auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, verify=S.VERIFY_SSL) as client:
            r = await client.request(method, f"{S.CHAT_BACKEND_URL}{path}", headers=headers, **kwargs)
    except httpx.HTTPError as e:
        logger.warning("Django unreachable for %s %s: %s", method, path, e)
        raise HTTPException(status_code=502, detail="Chat backend unreachable.")

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail="Chat backend rejected the request.")
    try:
        return r.json()
    except ValueError:
        return {}


@router.get("")
async def list_sessions(request: Request):
    auth_token = await _require_token(request)
    return await _django("GET", "/chat/sessions/", auth_token)


@router.get("/{session_id}")
async def get_session(session_id: str, request: Request):
    auth_token = await _require_token(request)
    return await _django("GET", f"/chat/sessions/{session_id}/", auth_token)


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request):
    auth_token = await _require_token(request)
    return await _django("DELETE", f"/chat/sessions/{session_id}/", auth_token)


@router.post("/log-turn")
async def log_turn(body: ChatTurnLogIn, request: Request):
    auth_token = await _require_token(request)
    return await _django("POST", "/chat/log-turn/", auth_token, json=body.model_dump())
