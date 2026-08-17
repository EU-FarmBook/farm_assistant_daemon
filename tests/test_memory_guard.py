"""
What may become a memory.

Both cases below were observed in the pilot, and both are the same mistake: the
agent treating what a question was ABOUT as a fact about the person asking. The
cost is asymmetric — an unwritten true fact costs one repetition, a written false
one shapes every later answer until the user hunts it down — so the guard fails
closed.
"""

import pytest

from app.config import Settings
from app.services import memory_guard


@pytest.fixture(autouse=True)
def keyed(monkeypatch):
    monkeypatch.setattr(memory_guard, "S", Settings(MISTRAL_API_KEY="k", _env_file=None))


def _verdict(monkeypatch, answer: str):
    """Stub the judge so these test OUR rules, not the model's mood."""
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": answer}}]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, *_a, **_k):
            return _Response()

    monkeypatch.setattr(memory_guard.httpx, "AsyncClient", lambda **_k: _Client())


# --- structural rejections, no model call needed --------------------------

def test_a_question_is_never_a_fact():
    assert memory_guard.structural_reject("Does the user farm in Italy?") is not None


def test_too_short_or_too_long_is_rejected():
    assert memory_guard.structural_reject("dairy") is not None
    assert memory_guard.structural_reject("x" * 500) is not None


def test_a_plausible_fact_passes_the_structural_check():
    assert memory_guard.structural_reject("The user farms dairy in the Netherlands.") is None


# --- the gate itself ------------------------------------------------------

@pytest.mark.asyncio
async def test_a_stated_fact_is_allowed(monkeypatch):
    _verdict(monkeypatch, "YES")
    allowed, _ = await memory_guard.is_supported_by_user(
        "The user farms dairy in the Netherlands.",
        "I run a dairy farm in the Netherlands.",
    )
    assert allowed


@pytest.mark.asyncio
async def test_the_topic_of_a_question_is_refused(monkeypatch):
    _verdict(monkeypatch, "NO")
    allowed, reason = await memory_guard.is_supported_by_user(
        "The user farms in Italy.",
        "What is pig manure used for in Italy?",
    )
    assert not allowed
    assert "not asserted" in reason


@pytest.mark.asyncio
async def test_validation_failure_refuses_rather_than_allows(monkeypatch):
    def explode(**_k):
        raise memory_guard.httpx.HTTPError("provider down")

    monkeypatch.setattr(memory_guard.httpx, "AsyncClient", explode)
    allowed, reason = await memory_guard.is_supported_by_user(
        "The user farms dairy in the Netherlands.", "I run a dairy farm here.",
    )
    # Fails closed: a false memory outlives an outage.
    assert not allowed
    assert "unavailable" in reason


@pytest.mark.asyncio
async def test_no_user_message_means_no_memory(monkeypatch):
    _verdict(monkeypatch, "YES")
    allowed, reason = await memory_guard.is_supported_by_user(
        "The user farms dairy in the Netherlands.", "",
    )
    assert not allowed
    assert "no user message" in reason


@pytest.mark.asyncio
async def test_without_a_provider_key_nothing_is_stored(monkeypatch):
    monkeypatch.setattr(memory_guard, "S", Settings(MISTRAL_API_KEY="", _env_file=None))
    allowed, _ = await memory_guard.is_supported_by_user(
        "The user farms dairy in the Netherlands.", "I farm dairy in the Netherlands.",
    )
    assert not allowed


def test_the_language_rule_is_stated_to_the_judge():
    # The other observed failure: one question in Hungarian became a stored
    # preference for Hungarian answers, which then fought the language rule.
    assert "Hungarian" in memory_guard._VERDICT_PROMPT
    assert "LANGUAGE" in memory_guard._VERDICT_PROMPT
