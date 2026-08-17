# app/routers/memory.py
"""
Settings-dialog surface: Personalization, Custom instructions, Memory.

The v3 shell is a copy of the v2 shell, so it calls the same five endpoints the
v2 dialog calls. They are served here as **pass-throughs to django_euf_admin**,
because that is where all of it already lives: tone, characteristics,
about_you, custom_instructions, memory_enabled and the memory notes are the same
rows v2 reads. Reshaping any of it here would make the two dialogs drift for no
reason.

The one exception is the memory summary. v2 generates it inside FA; v3 generates
it with a plain completion (see memory_service.generate_summary) rather than an
agent turn — summarising stored facts must not run a tool loop, enter the
transcript, or be able to trigger a retrieval.

`/memory/documents` is additional, not a replacement: it renders the same rows in
the two-document form (USER.md / MEMORY.md) the mneme design is built around.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas import MemoryDocumentPatchIn, MemoryDocumentsOut
from app.services import memory_service, suggestion_service
from app.services.auth_service import decode_token_email, resolve_user_uuid
from app.services.profile_registry import ProfileNotProvisioned, resolve_profile

logger = logging.getLogger("farm-assistant-hermes.memory-api")
router = APIRouter(prefix="/chatbot/api/users/me", tags=["Settings"])


async def _caller(request: Request) -> tuple[str, str, str]:
    auth_token = request.headers.get("Authorization", "")
    user_uuid = await resolve_user_uuid(auth_token) if auth_token else None
    if not user_uuid:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        profile = resolve_profile(user_uuid, email=decode_token_email(auth_token))
    except ProfileNotProvisioned:
        raise HTTPException(status_code=403, detail="This experimental assistant is limited to the pilot group.")
    return auth_token, user_uuid, profile


def _passthrough(status: int, body: dict) -> dict:
    if status >= 400:
        raise HTTPException(status_code=status, detail=body.get("detail") or "Request failed.")
    return body


# --- Personalization + custom instructions -------------------------------

@router.get("/settings")
async def get_settings(request: Request):
    auth_token, _, _ = await _caller(request)
    status, body = await memory_service.fetch_raw(auth_token, "/chat/user/settings/")
    return _passthrough(status, body)


@router.patch("/settings")
async def patch_settings(payload: dict, request: Request):
    auth_token, _, _ = await _caller(request)
    status, body = await memory_service.fetch_raw(
        auth_token, "/chat/user/settings/", method="PATCH", json_body=payload,
    )
    return _passthrough(status, body)


# --- Memory notes ---------------------------------------------------------

@router.get("/memory")
async def list_memory(request: Request, limit: int = 50):
    auth_token, _, _ = await _caller(request)
    status, body = await memory_service.fetch_raw(
        auth_token, "/chat/user/memory/", params={"limit": limit},
    )
    return _passthrough(status, body)


@router.delete("/memory/{note_id}")
async def delete_memory(note_id: int, request: Request):
    auth_token, user_uuid, _ = await _caller(request)
    ok = await memory_service.delete_note(auth_token, note_id)
    if ok:
        # Their memory just changed; the cached openers are now stale.
        suggestion_service.invalidate(user_uuid)
    if not ok:
        raise HTTPException(status_code=502, detail="Could not delete that memory right now.")
    return {"status": "ok"}


@router.delete("/memory")
async def forget_everything(request: Request):
    """
    Erase everything the assistant knows about the caller.

    Destructive and irreversible, so it is its own endpoint rather than a flag
    on another one — nothing should be able to trigger this as a side effect.
    """
    auth_token, user_uuid, _ = await _caller(request)
    result = await memory_service.forget_everything(auth_token)
    suggestion_service.invalidate(user_uuid)
    logger.info("Memory wipe for uuid=%s: %s", user_uuid, result)
    return {"status": "ok", **result}


@router.post("/memory/summary")
async def regenerate_summary(request: Request):
    """
    Regenerate and cache the memory overview.

    Falls back to whatever is already cached when generation is unavailable (no
    key, provider error): the dialog then shows a stale summary instead of an
    error, which is the better failure for a read-only nicety.
    """
    auth_token, _, _ = await _caller(request)
    mem = await memory_service.load(auth_token)

    summary = await memory_service.generate_summary(mem)
    if summary is None:
        return {"memory_summary": mem.memory_summary, "regenerated": False}

    await memory_service.save_summary(auth_token, summary)
    return {"memory_summary": summary, "regenerated": True}


@router.get("/suggestions")
async def get_suggestions(request: Request):
    """
    Opening prompts for an empty chat, generated from this user's memory.

    Cached for an hour per user, and falls back to a fixed EU-FarmBook set when
    there is nothing remembered yet — a first-time visitor should still be
    offered something sensible.
    """
    auth_token, user_uuid, _ = await _caller(request)
    suggestions, personalised = await suggestion_service.get_suggestions(auth_token, user_uuid)
    return {"status": "ok", "personalised": personalised, "suggestions": suggestions}


# --- The two-document view (v3 only) --------------------------------------

@router.get("/memory/documents", response_model=MemoryDocumentsOut)
async def get_documents(request: Request):
    auth_token, _, profile = await _caller(request)
    mem = await memory_service.load(auth_token)
    return MemoryDocumentsOut(profile=profile, documents=memory_service.render_documents(mem))


@router.patch("/memory/documents/USER.md")
async def patch_user_document(body: MemoryDocumentPatchIn, request: Request):
    """
    The user rewriting their own profile.

    Only USER.md is writable as a document. MEMORY.md is agent-authored: the user
    corrects it by deleting individual notes, so an accidental full-buffer
    overwrite cannot wipe the agent's memory in one keystroke.
    """
    auth_token, user_uuid, _ = await _caller(request)
    ok = await memory_service.save_about_you(auth_token, about_you=body.content)
    if ok:
        suggestion_service.invalidate(user_uuid)
    if not ok:
        raise HTTPException(status_code=502, detail="Could not save your profile right now.")
    return {"status": "ok"}
