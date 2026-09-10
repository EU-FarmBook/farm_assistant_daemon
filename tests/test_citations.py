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


# --- One turn at a time per profile --------------------------------------
#
# The register above is only coherent if a profile has ONE turn in flight. A
# profile is a user, and nothing stopped a user's second tab from starting a
# second turn into the same slot: the first stream then published the second
# turn's documents beside its own [n] citations, and whichever turn finished
# first revoked the other's tools. Silent mis-citation is exactly what this
# module exists to prevent, so the second turn is refused.

from app.services import tool_server  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_turns():
    tool_server._turns.clear()
    yield
    tool_server._turns.clear()


def _begin(profile="p1", token="Bearer a", uuid="uuid-a"):
    tool_server.begin_turn(profile, auth_token=token, user_uuid=uuid, user_message="q")


def test_a_second_concurrent_turn_is_refused():
    _begin()
    with pytest.raises(tool_server.TurnInProgress):
        _begin(token="Bearer b", uuid="uuid-a")


def test_the_first_turn_keeps_its_own_context_and_register():
    _begin(token="Bearer first")
    tool_server.search_eu_farmbook  # (tools resolve through _live_context)
    ctx = tool_server._live_context("p1")
    ctx.register([_src("doc-a")])
    with pytest.raises(tool_server.TurnInProgress):
        _begin(token="Bearer second")

    version, sources = tool_server.peek_sources("p1")
    assert [s.id for s in sources] == ["doc-a"]      # not replaced
    assert tool_server._live_context("p1").auth_token == "Bearer first"


def test_a_new_turn_is_allowed_once_the_previous_one_ends():
    _begin()
    tool_server.end_turn("p1")
    _begin(token="Bearer next")
    assert tool_server._live_context("p1").auth_token == "Bearer next"


def test_a_different_profile_is_never_blocked():
    _begin(profile="p1")
    _begin(profile="p2", uuid="uuid-b")
    assert tool_server._live_context("p1") is not None
    assert tool_server._live_context("p2") is not None


def test_an_idle_turn_does_not_lock_the_user_out():
    """
    A turn abandoned without its finally must not block the user forever — the
    TTL prune inside _live_context is what makes refusing safe rather than
    sticky.

    The context is aged directly rather than by patching the clock: TurnContext
    binds time.monotonic as a dataclass default_factory, so a patched clock
    moves the reader without moving the writer and every context looks expired.
    """
    _begin()
    tool_server._turns["p1"].touched -= tool_server._TURN_TTL_SECONDS + 1

    _begin(token="Bearer after-expiry")
    ctx = tool_server._live_context("p1")
    assert ctx is not None and ctx.auth_token == "Bearer after-expiry"


def test_a_long_running_turn_is_not_evicted_mid_stream():
    """
    The TTL is IDLE, not a deadline. It used to be measured from turn start, so
    a legitimately long turn — six agent iterations of a reasoning model, each
    with a retrieval — lost its own context while still streaming: its tools
    began answering "No active turn" and its citations vanished from the rail.
    """
    _begin()
    ctx = tool_server._live_context("p1")
    ctx.register([_src("doc-a")])

    # Simulate a turn that has run well past the TTL but is being touched, as
    # the streaming route touches it between tokens via peek_sources().
    for _ in range(3):
        tool_server._turns["p1"].started -= tool_server._TURN_TTL_SECONDS
        tool_server._turns["p1"].touched -= tool_server._TURN_TTL_SECONDS - 1
        assert tool_server.peek_sources("p1") is not None

    version, sources = tool_server.peek_sources("p1")
    assert [s.id for s in sources] == ["doc-a"]


def test_only_a_lack_of_access_expires_a_turn():
    _begin()
    tool_server._turns["p1"].touched -= tool_server._TURN_TTL_SECONDS + 1
    assert tool_server._live_context("p1") is None
    assert "p1" not in tool_server._turns          # and it is pruned, not left holding a JWT


# --- citations name documents, not chunks --------------------------------

def test_two_chunks_of_one_document_share_a_citation_number():
    """
    scout returns per-chunk ids ("<doc>::c0"), and the register keyed on `id` —
    so one document took two numbers and appeared twice in the source rail.
    """
    ctx = TurnContext(auth_token="t", user_uuid="u")
    url = "https://eufarmbook.eu/en/contributions/abc123"
    numbers = ctx.register([
        SourceItem(id="abc123::c0", url=url, title="Doc A"),
        SourceItem(id="abc123::c7", url=url, title="Doc A"),
        SourceItem(id="zzz::c0", url="https://eufarmbook.eu/en/contributions/zzz", title="Doc B"),
    ])
    assert numbers == [1, 1, 2]
    assert len(ctx.sources) == 2


def test_a_later_hop_reuses_the_document_number():
    ctx = TurnContext(auth_token="t", user_uuid="u")
    url = "https://eufarmbook.eu/en/contributions/abc123"
    ctx.register([SourceItem(id="abc123::c0", url=url, title="Doc A")])
    assert ctx.register([SourceItem(id="abc123::c9", url=url, title="Doc A")]) == [1]
    assert len(ctx.sources) == 1


def test_a_source_with_no_url_falls_back_to_the_document_half_of_the_id():
    ctx = TurnContext(auth_token="t", user_uuid="u")
    numbers = ctx.register([
        SourceItem(id="abc123::c0", title="Doc A"),
        SourceItem(id="abc123::c1", title="Doc A"),
    ])
    assert numbers == [1, 1]
