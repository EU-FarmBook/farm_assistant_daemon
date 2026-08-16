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
from app.schemas import ChatSessionCreateIn, ChatSessionPatchIn, ChatTurnLogIn, MessageFeedbackIn
from app.services import attachment_service
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


@router.post("")
async def create_session(body: ChatSessionCreateIn, request: Request):
    """
    Create an empty session.

    The shell calls this BEFORE streaming the first message — the stream URL is
    `/chats/{session_id}/message/stream`, so without a session there is nothing
    to stream to and the composer fails silently. Missing this endpoint is why a
    sent message appeared to do nothing at all.
    """
    auth_token = await _require_token(request)
    return await _django("POST", "/chat/sessions/", auth_token, json=body.model_dump())


@router.get("/{session_id}")
async def get_session(session_id: str, request: Request):
    auth_token = await _require_token(request)
    return await _django("GET", f"/chat/sessions/{session_id}/", auth_token)


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request):
    auth_token = await _require_token(request)
    return await _django("DELETE", f"/chat/sessions/{session_id}/", auth_token)


@router.patch("/{session_id}")
async def rename_session(session_id: str, body: ChatSessionPatchIn, request: Request):
    """Rename a session — the shell titles a chat after the first exchange."""
    auth_token = await _require_token(request)
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    return await _django("PATCH", f"/chat/sessions/{session_id}/", auth_token, json=payload)


@router.post("/{session_id}/messages/{message_id}/feedback")
async def message_feedback(session_id: str, message_id: int, body: MessageFeedbackIn,
                           request: Request):
    """Thumbs up/down on an answer."""
    auth_token = await _require_token(request)
    return await _django(
        "POST",
        f"/chat/sessions/{session_id}/message/{message_id}/feedback/",
        auth_token,
        json=body.model_dump(),
    )


@router.get("/{session_id}/attachments")
async def list_attachments(session_id: str, request: Request):
    """
    Documents attached to this session, for the chips above the composer.

    Answered rather than 404'd even when empty: the shell fetches this when
    opening a session, and an error here breaks loading the conversation.
    """
    auth_token = await _require_token(request)
    user_uuid = await resolve_user_uuid(auth_token)
    return {
        "status": "ok",
        "attachments": [
            {
                "doc_id": a.doc_id,
                "filename": a.filename,
                "mime_type": a.mime_type,
            }
            for a in attachment_service.for_session(session_id, user_uuid or "")
        ],
    }


@router.post("/log-turn")
async def log_turn(body: ChatTurnLogIn, request: Request):
    auth_token = await _require_token(request)
    return await _django("POST", "/chat/log-turn/", auth_token, json=body.model_dump())
