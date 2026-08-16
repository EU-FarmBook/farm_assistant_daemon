# app/services/attachment_service.py
"""
Document attachments: upload, extract, inject.

The agent never receives a file. A document is extracted to text at upload time
and that text is prepended to the user's message for the turns that reference
it — the same shape farm_assistant_um uses, and the reason no file ever needs to
cross into the agent container.

**Documents only, deliberately.** Images would need the vision model wired as
well; that is a separate piece of work and this one is useful without it. The
upload endpoint rejects anything unsupported with a specific message rather than
a generic failure.

**Storage is in-process, keyed by owner.** Consistent with the turn context and
the suggestion cache: this service is single-replica by design. What that costs
is honest and worth stating — attachments do not survive a restart, and a user
whose upload predates a deploy is told to upload it again rather than being
silently answered without their document. farm_assistant_um solves this with S3;
if v3 ever needs that, this module is the seam.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

from app.config import get_settings
from app.services.document_extractors import (
    LEGACY_UNSUPPORTED,
    SUPPORTED_DOCUMENT_TYPES,
    extract_text,
)

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.attachments")

_TTL_SECONDS = 24 * 3600.0
_MAX_DOCUMENTS = 2000


class AttachmentError(Exception):
    """Upload rejected — the message is written for the user, not the log."""


@dataclass
class Attachment:
    doc_id: str
    owner_uuid: str
    filename: str
    mime_type: str
    text: str
    session_id: Optional[str] = None
    created: float = field(default_factory=time.monotonic)


_documents: Dict[str, Attachment] = {}


def _prune() -> None:
    now = time.monotonic()
    for doc_id in [k for k, a in _documents.items() if now - a.created > _TTL_SECONDS]:
        _documents.pop(doc_id, None)
    while len(_documents) > _MAX_DOCUMENTS:
        oldest = min(_documents, key=lambda k: _documents[k].created)
        _documents.pop(oldest, None)


def store(
    *,
    owner_uuid: str,
    filename: str,
    payload: bytes,
    session_id: Optional[str] = None,
) -> Attachment:
    """Extract a document to text and keep it for this user."""
    suffix = Path(filename or "").suffix.lower()

    if suffix in LEGACY_UNSUPPORTED:
        raise AttachmentError(LEGACY_UNSUPPORTED[suffix])
    if suffix not in SUPPORTED_DOCUMENT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_TYPES))
        raise AttachmentError(f"Unsupported file type '{suffix or filename}'. Supported: {supported}.")
    if not payload:
        raise AttachmentError("That file is empty.")
    if len(payload) > S.ATTACHMENT_MAX_BYTES:
        limit_mb = S.ATTACHMENT_MAX_BYTES // (1024 * 1024)
        raise AttachmentError(f"That file is larger than {limit_mb} MB.")

    # Extract from a temp file and drop it immediately: the bytes are of no
    # further use once the text exists, and keeping them would mean holding
    # user documents on disk for no reason.
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / (Path(filename).name or f"upload{suffix}")
        path.write_bytes(payload)
        try:
            text = extract_text(path, filename)
        except Exception as e:  # noqa: BLE001 - extractors raise per-format errors
            logger.warning("Extraction failed for %s: %s", filename, e)
            raise AttachmentError("That file could not be read. Please check it opens correctly.") from e

    if not (text or "").strip():
        raise AttachmentError("No readable text was found in that file.")

    _prune()
    attachment = Attachment(
        doc_id=uuid.uuid4().hex,
        owner_uuid=owner_uuid,
        filename=Path(filename).name,
        mime_type=SUPPORTED_DOCUMENT_TYPES[suffix],
        text=text.strip()[:S.ATTACHMENT_MAX_CHARS],
        session_id=session_id,
    )
    _documents[attachment.doc_id] = attachment
    logger.info(
        "Stored attachment %s (%s, %d chars) for uuid=%s",
        attachment.doc_id, attachment.filename, len(attachment.text), owner_uuid,
    )
    return attachment


def get(doc_id: str, owner_uuid: str) -> Optional[Attachment]:
    """
    Fetch one attachment, enforcing ownership.

    The doc id is a random hex string, but it travels through the browser, so
    ownership is checked rather than assumed — an id is not a capability here.
    """
    attachment = _documents.get(doc_id)
    if not attachment or attachment.owner_uuid != owner_uuid:
        return None
    if time.monotonic() - attachment.created > _TTL_SECONDS:
        _documents.pop(doc_id, None)
        return None
    return attachment


def delete(doc_id: str, owner_uuid: str) -> bool:
    if get(doc_id, owner_uuid) is None:
        return False
    _documents.pop(doc_id, None)
    return True


def for_session(session_id: str, owner_uuid: str) -> List[Attachment]:
    return [
        a for a in _documents.values()
        if a.owner_uuid == owner_uuid and a.session_id == session_id
    ]


def build_context(doc_ids: List[str], owner_uuid: str) -> str:
    """
    Render the referenced documents as a block to prepend to the question.

    Labelled as user-provided and explicitly NOT as platform sources: the
    assistant cites EU-FarmBook material by number, and an uploaded file must
    never end up presented as if it came from the platform.
    """
    parts: List[str] = []
    for doc_id in doc_ids:
        attachment = get(doc_id.strip(), owner_uuid)
        if not attachment:
            continue
        parts.append(f"--- {attachment.filename} ---\n{attachment.text}")

    if not parts:
        return ""

    body = "\n\n".join(parts)
    return (
        "The user attached the following document(s) to this message. Treat them "
        "as material the USER provided, not as EU-FarmBook sources: answer from "
        "them where relevant, refer to them by filename, and never cite them with "
        "a [number] as though they came from the platform.\n\n"
        f"{body}\n\n--- end of attached document(s) ---"
    )


def reset() -> None:
    """Test helper."""
    _documents.clear()
