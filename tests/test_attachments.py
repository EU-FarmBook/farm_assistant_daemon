"""
Document attachments.

Two properties matter beyond "it extracts text": a user can only reach their own
uploads, and an attached document is never presented to the agent as an
EU-FarmBook source. The second one protects citations — the whole point of the
assistant is that [1] means a platform document.
"""

import pytest

from app.config import Settings
from app.services import attachment_service
from app.services.attachment_service import AttachmentError

OWNER = "uuid-a"
OTHER = "uuid-b"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(attachment_service, "S", Settings(_env_file=None))
    attachment_service.reset()
    yield
    attachment_service.reset()


def _store(text=b"Cover crops improve soil structure.", name="notes.txt", owner=OWNER, session=None):
    return attachment_service.store(
        owner_uuid=owner, filename=name, payload=text, session_id=session,
    )


def test_txt_upload_extracts_text():
    a = _store()
    assert a.filename == "notes.txt"
    assert "Cover crops" in a.text
    assert a.mime_type == "text/plain"


def test_legacy_formats_get_an_actionable_message():
    with pytest.raises(AttachmentError) as exc:
        _store(name="report.doc")
    # "unsupported file" tells the user nothing they can act on.
    assert ".docx" in str(exc.value)


def test_unsupported_type_is_rejected():
    with pytest.raises(AttachmentError):
        _store(name="archive.zip")


def test_empty_file_is_rejected():
    with pytest.raises(AttachmentError):
        _store(text=b"", name="empty.txt")


def test_oversized_file_is_rejected(monkeypatch):
    monkeypatch.setattr(attachment_service, "S", Settings(ATTACHMENT_MAX_BYTES=10, _env_file=None))
    with pytest.raises(AttachmentError) as exc:
        _store(text=b"x" * 50)
    assert "MB" in str(exc.value)


def test_another_user_cannot_read_or_delete_it():
    a = _store()
    # The doc id travels through the browser; it is an identifier, not a
    # capability, so ownership is checked on every read.
    assert attachment_service.get(a.doc_id, OTHER) is None
    assert attachment_service.delete(a.doc_id, OTHER) is False
    assert attachment_service.get(a.doc_id, OWNER) is not None


def test_context_marks_documents_as_user_provided_not_sources():
    a = _store()
    context = attachment_service.build_context([a.doc_id], OWNER)
    assert "notes.txt" in context
    assert "Cover crops" in context
    # The agent must not cite an upload as though it were platform material.
    assert "not as EU-FarmBook sources" in context
    assert "never cite them with" in context


def test_context_ignores_documents_owned_by_someone_else():
    mine = _store()
    theirs = _store(name="theirs.txt", owner=OTHER)
    context = attachment_service.build_context([mine.doc_id, theirs.doc_id], OWNER)
    assert "notes.txt" in context
    assert "theirs.txt" not in context


def test_unknown_doc_ids_yield_no_context():
    assert attachment_service.build_context(["nope"], OWNER) == ""
    assert attachment_service.build_context([], OWNER) == ""


def test_session_listing_is_scoped_to_owner_and_session():
    a = _store(session="s1")
    _store(name="other.txt", session="s2")
    _store(name="theirs.txt", owner=OTHER, session="s1")
    listed = attachment_service.for_session("s1", OWNER)
    assert [x.doc_id for x in listed] == [a.doc_id]
