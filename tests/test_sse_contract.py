"""
The SSE payload contract with the v3 shell.

Learned the hard way: the shell appends `token` payloads to the answer WITHOUT
parsing them, and reads the completed answer from `final.text`. Sending
`{"text": ...}` per token rendered literal JSON in the chat window, and
`final.answer` was silently ignored because the shell only looks for `text`.

These assert the shapes rather than the plumbing, because the plumbing was never
the part that broke.
"""

import inspect
import re

from app.routers import ask


def _source() -> str:
    # The turn implementation moved out of the route when the
    # non-streaming door was added; both doors consume this one generator.
    return inspect.getsource(ask._turn_events)


def test_token_is_emitted_as_a_bare_string():
    src = _source()
    # The shell does `assistantText + event.data` — no JSON.parse. Anything
    # other than the raw delta shows up as literal text in the bubble.
    assert 'yield ("token", delta)' in src
    assert 'yield ("token", {' not in src
    # And the framing keeps a str payload untouched.
    framing = inspect.getsource(ask.stream_message)
    assert 'payload = data if isinstance(data, str) else json.dumps' in framing


def test_final_uses_the_key_the_shell_reads():
    src = _source()
    assert 'yield ("final", {"text": answer})' in src
    # `answer` was the original key; the shell ignores it and leaves the raw
    # streamed text in place, which looks like a rendering bug rather than a
    # contract mismatch.
    assert '"answer": answer' not in src


def test_structured_events_stay_json():
    src = _source()
    # Every event but `token` must be a dict/list the framing will JSON-encode.
    seq = _emit_sequence()
    for event in ("sources", "grounding", "timing", "done", "status", "app_error"):
        assert event in seq, f"{event} is no longer emitted"
    assert 'yield ("token", delta)' in src          # the one bare string


def test_emit_passes_strings_through_untouched():
    # Guards the helper itself: if it ever JSON-encodes strings, every token
    # regresses to literal JSON in the chat.
    src = inspect.getsource(ask.stream_message)
    assert "isinstance(data, str)" in src


def _emit_sequence() -> list:
    """Event names in source order — emit() calls are formatted across lines."""
    return re.findall(r'yield \(\s*"(\w+)"', _source())


def test_every_terminal_path_ends_with_done():
    """
    A stream that ends without `done` is a clean EOF, which the EventSource spec
    tells a client to RECONNECT on — so each failure silently re-ran a whole
    billed agent turn on a loop. `done` used to appear on the success path only.
    """
    seq = _emit_sequence()
    assert seq.count("app_error") == 3          # empty answer, agent down, catch-all
    assert seq.count("done") == 4               # those three, plus success
    src = _source()
    assert 'yield ("done", {"ok": True})' in src
    assert src.count('"ok": False') == 3


def test_app_error_carries_a_stable_code():
    """A client should branch on a code, not on an English sentence."""
    src = _source()
    for code in ("empty_answer", "agent_unavailable", "internal_error"):
        assert f'"code": "{code}"' in src


def test_a_terminal_done_follows_each_app_error():
    """Ordering matters: the stream's end must come WITH the error, not before."""
    seq = _emit_sequence()
    for i, name in enumerate(seq):
        if name == "app_error":
            assert seq[i + 1] == "done", f"app_error at {i} is followed by {seq[i + 1]}"
