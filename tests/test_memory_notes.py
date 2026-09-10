"""
Which stored notes actually reach the prompt.

The bug these exist to prevent: the eligibility filter required a `confidence`
field on every row, defaulting a missing one to 0 against a 0.6 threshold —
while add_note() posts `{"note_text": ...}` and never sends a confidence. So
every note the agent wrote was silently invisible to its own prompt, and
GET /users/me/memory still listed them, so nothing looked broken.

Three things read this one list, and all three failed together: the Background
block, the [M<n>] marker -> id map forget_about_user needs, and the existing
notes find_superseded consolidates against.
"""

import pytest

from app.config import Settings
from app.services import memory_service, tool_server
from app.services.memory_service import UserMemory, usable_notes


def _note(nid, text="The user farms dairy in Brittany", **extra):
    """A row exactly as add_note() writes it, plus the id Django assigns."""
    return {"id": nid, "note_text": text, **extra}


# --- eligibility ---------------------------------------------------------

def test_a_note_without_a_confidence_field_is_usable():
    """The regression. This is the shape add_note() actually produces."""
    assert len(usable_notes(UserMemory(notes=[_note(1)]))) == 1


def test_an_explicitly_low_confidence_note_is_dropped():
    """The filter still does its job when Django expresses an opinion."""
    assert usable_notes(UserMemory(notes=[_note(1, confidence=0.2)])) == []


def test_a_high_confidence_note_is_kept_under_either_field_name():
    assert len(usable_notes(UserMemory(notes=[_note(1, confidence=0.9)]))) == 1
    assert len(usable_notes(UserMemory(notes=[_note(2, confidence_score=0.9)]))) == 1


def test_a_zero_confidence_note_is_dropped_but_a_null_one_is_not():
    """None is "no opinion"; 0.0 is an opinion."""
    assert usable_notes(UserMemory(notes=[_note(1, confidence=0.0)])) == []
    assert len(usable_notes(UserMemory(notes=[_note(2, confidence=None)]))) == 1


def test_an_unparseable_confidence_does_not_silently_delete_the_note():
    assert len(usable_notes(UserMemory(notes=[_note(1, confidence="n/a")]))) == 1


def test_an_empty_note_is_still_skipped():
    assert usable_notes(UserMemory(notes=[_note(1, text="   ")])) == []


def test_the_prompt_budget_still_applies_after_filtering():
    notes = [_note(i, text=f"fact {i}") for i in range(20)]
    assert len(usable_notes(UserMemory(notes=notes))) == memory_service._MAX_PROMPT_NOTES


# --- the three consumers -------------------------------------------------

def test_the_background_block_carries_a_confidence_less_note(monkeypatch):
    monkeypatch.setattr(memory_service, "S", Settings(_env_file=None))
    block = memory_service.render_memory_block(UserMemory(notes=[_note(1)]))
    assert "Brittany" in block
    assert "[M1]" in block          # the marker the agent needs to forget by


def test_the_marker_map_is_populated_so_forgetting_works():
    """
    ask.py feeds usable_notes() into set_notes(); with everything filtered the
    agent got no markers, so forget_about_user could never name a note.
    """
    tool_server._turns.clear()
    tool_server.begin_turn("p1", auth_token="Bearer t", user_uuid="uuid-a", user_message="hi")
    tool_server.set_notes("p1", usable_notes(UserMemory(notes=[_note(41), _note(42, text="grows maize")])))
    ctx = tool_server._live_context("p1")
    assert ctx.note_ids == [41, 42]
    assert ctx.note_texts[1] == "grows maize"
    tool_server.end_turn("p1")


def test_consolidation_has_something_to_supersede_against():
    """
    find_superseded() compares a new fact to ctx.note_texts. Empty means every
    write appends, which is how a user ends up recorded in two countries at once.
    """
    assert memory_service._usable_notes(UserMemory(notes=[_note(1)])) == [
        "The user farms dairy in Brittany"
    ]


def test_all_notes_filtered_is_logged_loudly(caplog):
    """The signal that was missing while this failed silently."""
    with caplog.at_level("WARNING"):
        usable_notes(UserMemory(notes=[_note(1, confidence=0.1)]))
    assert any("will not reach the prompt" in r.message for r in caplog.records)


# --- The USER.md document round-trip -------------------------------------

def test_user_md_contains_only_the_field_that_patch_writes():
    """
    It used to concatenate about_you AND custom_instructions, while PATCH writes
    the whole buffer back into about_you alone — so every save appended the
    instructions to the profile and left them in place too. The two are not
    interchangeable: render_memory_block calls custom_instructions style ("only
    tone, format and level of detail") and about_you authoritative knowledge, so
    the round trip promoted style text to fact.
    """
    mem = UserMemory(about_you="## Farm\n40 ha near Rennes", custom_instructions="Answer briefly.")
    doc = next(d for d in memory_service.render_documents(mem) if d.name == "USER.md")
    assert doc.content == "## Farm\n40 ha near Rennes"
    assert "briefly" not in doc.content
    assert doc.char_count == len(doc.content)


def test_the_round_trip_is_stable():
    """GET, then PATCH what you were given, must not change the document."""
    mem = UserMemory(about_you="I farm dairy.", custom_instructions="Be terse.")
    first = next(d for d in memory_service.render_documents(mem) if d.name == "USER.md")
    # PATCH writes first.content into about_you; custom_instructions is untouched.
    again = next(
        d for d in memory_service.render_documents(
            UserMemory(about_you=first.content, custom_instructions="Be terse.")
        ) if d.name == "USER.md"
    )
    assert again.content == first.content
    assert again.char_count == first.char_count


# --- unread is not the same as empty -------------------------------------

async def test_a_failed_read_is_marked_unloaded(monkeypatch):
    """
    load() fails soft to an empty profile so a Django blip costs personalization
    rather than the answer. But an empty profile and an unreachable backend must
    be distinguishable, or a compare-and-swap ends up comparing against a guess.
    """
    import httpx

    from app.config import Settings

    monkeypatch.setattr(memory_service, "S", Settings(CHAT_BACKEND_URL="https://x", _env_file=None))

    class _Boom:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(memory_service.httpx, "AsyncClient", _Boom)
    mem = await memory_service.load("Bearer t")
    assert mem.loaded is False
    assert mem.notes == []


async def test_a_non_200_is_also_unloaded_and_logged(monkeypatch, caplog):
    """A 4xx/5xx used to look exactly like a user with no profile, silently."""
    from app.config import Settings

    monkeypatch.setattr(memory_service, "S", Settings(CHAT_BACKEND_URL="https://x", _env_file=None))

    class _Resp:
        status_code = 502

        def json(self):
            return {}

    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(memory_service.httpx, "AsyncClient", _Client)
    with caplog.at_level("WARNING"):
        mem = await memory_service.load("Bearer t")
    assert mem.loaded is False
    assert any("502" in r.getMessage() for r in caplog.records)


def test_a_genuinely_empty_profile_is_still_loaded():
    """The distinction cuts both ways: no memory is not a failure."""
    assert UserMemory().loaded is True
