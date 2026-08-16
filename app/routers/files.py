# app/routers/files.py
"""
Document attachments for the composer's `+` button.

Speaks the same endpoints the v2 shell calls, so the copied UI works unchanged:
upload, delete, view-url, and the per-session listing. Images are deliberately
absent — `/files/image` is not implemented, and the composer's image path will
fail with a clear message rather than half-working.
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.services import attachment_service
from app.services.attachment_service import AttachmentError
from app.services.auth_service import decode_token_email, resolve_user_uuid
from app.services.profile_registry import ProfileNotProvisioned, resolve_profile

logger = logging.getLogger("farm-assistant-hermes.files")
router = APIRouter(prefix="/chatbot/api/files", tags=["Files"])


async def _owner(request: Request) -> str:
    """The verified uuid that will own the upload. Ownership is the access rule."""
    auth_token = request.headers.get("Authorization", "")
    user_uuid = await resolve_user_uuid(auth_token) if auth_token else None
    if not user_uuid:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        resolve_profile(user_uuid, email=decode_token_email(auth_token))
    except ProfileNotProvisioned:
        raise HTTPException(status_code=403, detail="This experimental assistant is limited to the pilot group.")
    return user_uuid


@router.post("/document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_uuid: Optional[str] = Form(default=None),
):
    owner = await _owner(request)
    payload = await file.read()

    try:
        attachment = attachment_service.store(
            owner_uuid=owner,
            filename=file.filename or "upload",
            payload=payload,
            session_id=session_uuid,
        )
    except AttachmentError as e:
        # 400 with the message as written: these are all user-actionable
        # ("save as .docx", "larger than 15 MB"), and burying them behind a
        # generic failure is what makes an upload button feel broken.
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "ok",
        "doc_id": attachment.doc_id,
        "filename": attachment.filename,
        "mime_type": attachment.mime_type,
        "chars": len(attachment.text),
    }


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    owner = await _owner(request)
    if not attachment_service.delete(doc_id, owner):
        raise HTTPException(status_code=404, detail="That attachment is no longer available.")
    return {"status": "ok"}


@router.get("/{doc_id}/url")
async def attachment_url(doc_id: str, request: Request):
    """
    v2 returns a presigned S3 link here. v3 keeps no file — only extracted
    text — so there is nothing to link to. Say so plainly instead of returning
    a URL that would 404 in a new tab.
    """
    owner = await _owner(request)
    if attachment_service.get(doc_id, owner) is None:
        raise HTTPException(status_code=404, detail="That attachment is no longer available.")
    raise HTTPException(
        status_code=501,
        detail="Preview is not available for attachments in this experimental assistant.",
    )
