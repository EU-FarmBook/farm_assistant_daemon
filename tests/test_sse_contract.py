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

from app.routers import ask


def _source() -> str:
    return inspect.getsource(ask.stream_message)


def test_token_is_emitted_as_a_bare_string():
    src = _source()
    # The shell does `assistantText + event.data` — no JSON.parse. Anything
    # other than the raw delta shows up as literal text in the bubble.
    assert 'emit("token", delta)' in src
    assert 'emit("token", {' not in src


def test_final_uses_the_key_the_shell_reads():
    src = _source()
    assert 'emit("final", {"text": answer})' in src
    # `answer` was the original key; the shell ignores it and leaves the raw
    # streamed text in place, which looks like a rendering bug rather than a
    # contract mismatch.
    assert '"answer": answer' not in src


def test_structured_events_stay_json():
    src = _source()
    # `app_error` is emitted multi-line in the except blocks, so match loosely.
    for event in ("sources", "grounding", "timing", "done", "status"):
        assert f'emit("{event}"' in src
    assert '"app_error"' in src


def test_emit_passes_strings_through_untouched():
    # Guards the helper itself: if it ever JSON-encodes strings, every token
    # regresses to literal JSON in the chat.
    src = inspect.getsource(ask)
    assert "if not isinstance(data, str):" in src
