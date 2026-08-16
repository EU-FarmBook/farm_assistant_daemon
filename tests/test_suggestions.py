"""
Memory-derived opening suggestions.

The failure that matters is not "no suggestions" — it is suggestions that leak
something they should not, or that quietly become a generic list while claiming
to be personal. Hence the `personalised` flag in the response.
"""

import pytest

from app.config import Settings
from app.services import memory_service, suggestion_service


@pytest.fixture(autouse=True)
def clean():
    suggestion_service.reset()
    yield
    suggestion_service.reset()


def _mem(**kwargs) -> memory_service.UserMemory:
    return memory_service.UserMemory(**kwargs)


@pytest.mark.asyncio
async def test_defaults_when_nothing_is_remembered(monkeypatch):
    monkeypatch.setattr(suggestion_service, "S", Settings(MISTRAL_API_KEY="k", _env_file=None))

    async def empty(_token):
        return _mem()

    monkeypatch.setattr(memory_service, "load", empty)
    suggestions, personalised = await suggestion_service.get_suggestions("t", "uuid-a")

    assert personalised is False
    assert suggestions == suggestion_service._DEFAULTS


@pytest.mark.asyncio
async def test_no_provider_key_does_not_spend_a_call(monkeypatch):
    monkeypatch.setattr(suggestion_service, "S", Settings(MISTRAL_API_KEY="", _env_file=None))

    async def remembered(_token):
        return _mem(about_you="I farm dairy in Brittany.")

    monkeypatch.setattr(memory_service, "load", remembered)

    def explode(*_a, **_k):
        raise AssertionError("must not call the provider without a key")

    monkeypatch.setattr(suggestion_service.httpx, "AsyncClient", explode)
    suggestions, personalised = await suggestion_service.get_suggestions("t", "uuid-a")
    assert personalised is False
    assert suggestions == suggestion_service._DEFAULTS


@pytest.mark.asyncio
async def test_memory_disabled_falls_back(monkeypatch):
    monkeypatch.setattr(suggestion_service, "S", Settings(MISTRAL_API_KEY="k", _env_file=None))

    async def paused(_token):
        return _mem(about_you="I farm dairy in Brittany.", memory_enabled=False)

    monkeypatch.setattr(memory_service, "load", paused)
    _, personalised = await suggestion_service.get_suggestions("t", "uuid-a")
    # Memory off must mean memory off — including for the suggestions that
    # would otherwise advertise what the agent knows.
    assert personalised is False


def test_parses_a_fenced_json_reply():
    parsed = suggestion_service._parse(
        '```json\n[{"title": "Plan grazing", "subtitle": "for a wet spring", '
        '"prompt": "How should I plan grazing after a wet spring in Brittany?"}]\n```'
    )
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Plan grazing"


def test_parse_drops_incomplete_entries_and_caps_at_three():
    parsed = suggestion_service._parse(
        '[{"title": "A", "prompt": "one"}, {"title": "", "prompt": "no title"},'
        ' {"prompt": "no title key"}, {"title": "B", "prompt": "two"},'
        ' {"title": "C", "prompt": "three"}, {"title": "D", "prompt": "four"}]'
    )
    assert [p["title"] for p in parsed] == ["A", "B", "C"]


def test_parse_survives_junk():
    assert suggestion_service._parse("sorry, I can't do that") == []
    assert suggestion_service._parse("") == []
