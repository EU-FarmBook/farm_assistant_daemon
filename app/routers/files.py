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

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile

from app.schemas import DocumentExportIn
from app.services import attachment_service
from app.services.document_export_service import content_disposition, generate_document
from app.services.attachment_service import AttachmentError
from app.services.auth_service import decode_token_email, resolve_user_uuid
from app.services.profile_registry import ProfileNotProvisioned, resolve_profile

from app.config import get_settings

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.files")

# 256 KiB: big enough that a 15 MB upload is ~60 reads, small enough that the
# overshoot past the limit before we refuse is negligible.
_UPLOAD_CHUNK_BYTES = 256 * 1024
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

    # Read in chunks and stop at the limit, instead of `await file.read()`.
    # attachment_service.store() also checks the size, but it could only check
    # AFTER the whole body was already resident: a single oversized upload put
    # its full length in the adapter's memory before being told it was too big,
    # and the adapter container has no memory limit while the agent has 4 GB.
    # Reading to the cap bounds that to ATTACHMENT_MAX_BYTES + one chunk.
    max_bytes = S.ATTACHMENT_MAX_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"That file is larger than {max_bytes // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    payload = b"".join(chunks)

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


@router.post("/export")
async def export_document(body: DocumentExportIn, request: Request):
    """
    Turn an assistant answer into a downloadable document.

    Copied wholesale from farm_assistant_um — same formats, same renderer, same
    source appendix — so a PDF from v3 is indistinguishable from a v2 one. There
    is nothing agent-specific about turning markdown into a file, and a second
    implementation would only be a second set of layout bugs.
    """
    await _owner(request)

    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Export content is required.")

    try:
        document = generate_document(
            title=body.title.strip() or "Farm Assistant response",
            content=content,
            export_format=body.format,
            sources=[source.model_dump(exclude_none=True) for source in body.sources],
        )
    except ImportError as error:
        # A format whose optional library is missing degrades to a clear message
        # rather than a 500 the user cannot act on.
        raise HTTPException(
            status_code=503, detail=f"{body.format.upper()} export is not installed.",
        ) from error
    except Exception as error:
        logger.exception("Export failed for format=%s", body.format)
        raise HTTPException(
            status_code=500, detail=f"Unable to generate {body.format.upper()} document.",
        ) from error

    return Response(
        content=document.payload,
        media_type=document.media_type,
        headers={
            "Content-Disposition": content_disposition(document.filename),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
