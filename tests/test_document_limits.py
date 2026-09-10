"""
Bounds on attachment handling.

Both of these are about a single-replica service with no memory limit in
compose: whatever one upload can take, it takes from every in-flight stream.
"""

import zipfile

import pytest

from app.services import document_extractors
from app.services.document_extractors import MAX_UNCOMPRESSED_BYTES, extract_text


def test_a_small_file_declaring_a_huge_expansion_is_refused(tmp_path):
    """
    The per-format caps all apply to text ALREADY extracted, which is too late:
    openpyxl/python-docx expand the zip container before this module sees a
    character. Measured on the original: 1.9 GB RSS from a 668 KB file.
    """
    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"\0" * (MAX_UNCOMPRESSED_BYTES + 1024))

    assert bomb.stat().st_size < 2 * 1024 * 1024          # tiny on disk
    with pytest.raises(RuntimeError, match="extraction limit"):
        extract_text(bomb, "bomb.docx")


def test_the_check_reads_only_the_central_directory(tmp_path, monkeypatch):
    """It must not decompress to decide — that would be the bomb doing its work."""
    bomb = tmp_path / "b.xlsx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/sheet1.xml", b"\0" * (MAX_UNCOMPRESSED_BYTES + 1024))

    def _explode(*a, **k):
        raise AssertionError("read() must not be called to size the archive")

    monkeypatch.setattr(zipfile.ZipFile, "read", _explode)
    with pytest.raises(RuntimeError, match="extraction limit"):
        extract_text(bomb, "b.xlsx")


def test_an_ordinary_document_is_unaffected(tmp_path):
    ok = tmp_path / "fine.txt"
    ok.write_text("Cover crops fix nitrogen.", encoding="utf-8")
    assert "nitrogen" in extract_text(ok, "fine.txt")


def test_a_corrupt_archive_becomes_a_user_facing_error(tmp_path, monkeypatch):
    """
    The extractors raise their own per-format exceptions (python-docx raises
    PackageNotFoundError here), so the contract that matters is the service
    layer's: anything unreadable becomes an AttachmentError, which the route
    turns into a 400 the user can act on — never a 500.
    """
    from app.config import Settings
    from app.services import attachment_service

    monkeypatch.setattr(attachment_service, "S", Settings(_env_file=None))
    broken = b"PK\x03\x04" + b"garbage" * 100
    with pytest.raises(attachment_service.AttachmentError, match="could not be read"):
        attachment_service.store(owner_uuid="u", filename="broken.docx", payload=broken)


def test_the_upload_route_reads_in_bounded_chunks():
    """
    The handler used to `await file.read()` the whole body and only then check
    the size, so an oversized upload was fully resident before being refused.
    """
    import inspect

    from app.routers import files

    # Comments are stripped first: the explanation of this very fix quotes the
    # old call, which a naive substring check would trip over.
    src = inspect.getsource(files.upload_document)
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "await file.read(_UPLOAD_CHUNK_BYTES)" in code
    assert "await file.read()" not in code
    assert "status_code=413" in src
    assert files._UPLOAD_CHUNK_BYTES <= 1024 * 1024


# --- exports keep the whole answer ---------------------------------------

_ANSWER = (
    "Cover crops raise soil nitrogen in three ways.\n\n"
    "| Practice | Benefit |\n|---|---|\n| Clover | fixes N |\n| Vetch | fixes N |\n\n"
    "Sow early for maximum capture."
)


def _readable(payload: bytes, fmt: str) -> str:
    import io
    import zipfile

    if fmt == "csv":
        return payload.decode("utf-8", "replace")
    if fmt == "pdf":
        from pypdf import PdfReader

        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(payload)).pages)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        return " ".join(
            z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".xml")
        )


@pytest.mark.parametrize("fmt", ["csv", "xlsx", "docx", "pptx", "pdf"])
def test_an_answer_with_a_table_keeps_its_prose(fmt):
    """
    Every exporter was `if table: render the table ELSE render the prose`, so
    the normal shape of an answer — prose around a comparison table, which
    SOUL.md explicitly asks for — exported as the bare table.
    """
    from app.services.document_export_service import generate_document

    body = _readable(generate_document(title="Answer", content=_ANSWER, export_format=fmt, sources=[]).payload, fmt)
    assert "three ways" in body, f"{fmt} dropped the leading prose"
    assert "Sow early" in body, f"{fmt} dropped the trailing prose"
    assert "Clover" in body, f"{fmt} dropped the table"


def test_a_csv_still_opens_as_a_grid():
    """The table stays first: prose above it would shift the header row."""
    import csv as _csv
    import io

    from app.services.document_export_service import generate_document

    text = generate_document(title="A", content=_ANSWER, export_format="csv", sources=[]).payload.decode("utf-8")
    rows = list(_csv.reader(io.StringIO(text.lstrip("﻿"))))
    assert rows[0] == ["Practice", "Benefit"]
    assert rows[1] == ["Clover", "fixes N"]


def test_an_answer_with_no_table_is_unchanged():
    from app.services.document_export_service import generate_document

    text = generate_document(title="A", content="Just prose.\n\nTwo paragraphs.",
                             export_format="csv", sources=[]).payload.decode("utf-8")
    assert "Just prose." in text and "Two paragraphs." in text


# --- the download header survives 24 languages ---------------------------

@pytest.mark.parametrize("title", [
    "Cover crops", "Καλλιέργειες εδάφους", "Zöldtrágya", "Pěstování", "全部",
])
def test_the_attachment_header_is_latin_1_safe(title):
    """
    Starlette encodes headers as latin-1 and _safe_filename keeps unicode word
    characters, so a Greek/Polish/Czech/Hungarian title raised
    UnicodeEncodeError and the export 500'd.
    """
    from app.services.document_export_service import content_disposition, generate_document

    doc = generate_document(title=title, content="text", export_format="csv", sources=[])
    header = content_disposition(doc.filename)
    header.encode("latin-1")                      # what Starlette will do
    assert "filename*=UTF-8''" in header          # the real name still travels


def test_a_wholly_non_ascii_title_gets_a_usable_fallback_name():
    """The ASCII fallback must not degenerate to the bare extension."""
    from app.services.document_export_service import content_disposition

    header = content_disposition("Καλλιέργειες.csv")
    assert 'filename="farm-assistant-response.csv"' in header
