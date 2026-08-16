"""
Tests for the turn citation register.

The defect these exist to prevent: an agent that searches twice gets two
passages both numbered [1], cites one, and the UI renders the other. Silent
mis-citation is worse for a source-cited assistant than no citation at all, so
numbering must be turn-global and the UI must see every hop.
"""

import pytest

from app.schemas import SourceItem
from app.services.tool_server import TurnContext


def _src(sid: str, title: str = "") -> SourceItem:
    return SourceItem(id=sid, title=title or f"Doc {sid}", url=f"https://x/{sid}")


@pytest.fixture
def ctx() -> TurnContext:
    return TurnContext(auth_token="Bearer t", user_uuid="uuid-a")


def test_first_hop_numbers_from_one(ctx):
    assert ctx.register([_src("a"), _src("b")]) == [1, 2]
    assert ctx.version == 1


def test_second_hop_continues_the_numbering(ctx):
    ctx.register([_src("a"), _src("b")])
    # THE bug: without a turn-global register these would come back [1, 2] again.
    assert ctx.register([_src("c"), _src("d")]) == [3, 4]
    assert [s.id for s in ctx.sources] == ["a", "b", "c", "d"]


def test_repeat_source_keeps_its_original_number(ctx):
    ctx.register([_src("a"), _src("b")])
    # Overlapping results across hops are normal; citing [1] must keep meaning
    # the same document, and the source rail must not list it twice.
    assert ctx.register([_src("b"), _src("e")]) == [2, 3]
    assert [s.id for s in ctx.sources] == ["a", "b", "e"]


def test_version_advances_per_hop_so_the_ui_is_re_emitted(ctx):
    ctx.register([_src("a")])
    v1 = ctx.version
    ctx.register([_src("b")])
    assert ctx.version > v1


def test_empty_hop_still_bumps_version_without_adding_sources(ctx):
    # A search that found nothing is still an event: the stream needs to publish
    # "no sources" rather than leaving the client waiting.
    assert ctx.register([]) == []
    assert ctx.version == 1
    assert ctx.sources == []


def test_sources_are_matched_by_url_when_id_is_absent(ctx):
    a = SourceItem(url="https://x/1", title="One")
    again = SourceItem(url="https://x/1", title="One")
    ctx.register([a])
    assert ctx.register([again]) == [1]
    assert len(ctx.sources) == 1
