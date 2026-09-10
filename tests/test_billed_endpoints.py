"""
The endpoints that spend on the provider, and who is allowed to.

/chatbot/api/follow-ups, /chatbot/api/export-intent and
/chatbot/api/chats/<id>/title each make a paid completion, and each required
nothing but a resolvable uuid — no pilot gate, no rate limit. With a closed
roster that meant a user who gets a hard 403 from the streaming endpoint could
still POST here in a loop and spend.

Two properties: they refuse to SPEND, and they still do not fail (each sits
beside an answer that was already delivered).
"""

import pytest

from app.config import Settings
from app.routers import _access
from app.services import profile_registry, rate_limit


@pytest.fixture(autouse=True)
def _clean():
    rate_limit.reset()
    yield
    rate_limit.reset()


class _Req:
    def __init__(self, token="Bearer t"):
        self.headers = {"Authorization": token} if token else {}


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def _wire(monkeypatch, *, uuid="uuid-a", open_access=True, **limits):
    s = _settings(HERMES_OPEN_ACCESS=open_access, HERMES_PILOT_UUIDS="", **limits)
    monkeypatch.setattr(profile_registry, "get_settings", lambda: s)
    monkeypatch.setattr(rate_limit, "get_settings", lambda: s)

    async def _uuid(_token):
        return uuid

    monkeypatch.setattr(_access, "resolve_user_uuid", _uuid)
    monkeypatch.setattr(_access, "decode_token_email", lambda _t: None)
    monkeypatch.setattr(
        _access, "resolve_profile",
        lambda u, email=None: profile_registry.resolve_profile(u, email=email, provision=False),
    )
    return s


async def test_an_authenticated_pilot_user_may_spend(monkeypatch):
    _wire(monkeypatch)
    _, uuid, refusal = await _access.spend_allowed(_Req())
    assert (uuid, refusal) == ("uuid-a", None)


async def test_no_token_is_refused_without_spending(monkeypatch):
    _wire(monkeypatch)

    async def _none(_t):
        return None

    monkeypatch.setattr(_access, "resolve_user_uuid", _none)
    _, uuid, refusal = await _access.spend_allowed(_Req(token=None))
    assert (uuid, refusal) == (None, "unauthenticated")


async def test_a_user_off_the_roster_is_refused_without_spending(monkeypatch):
    """The hole: 403 on chat, but these endpoints would still have spent."""
    _wire(monkeypatch, open_access=False)
    _, _, refusal = await _access.spend_allowed(_Req())
    assert refusal == "not_in_pilot"


async def test_the_allowance_runs_out(monkeypatch):
    _wire(monkeypatch, RATE_LIMIT_TURNS_PER_MIN=2)
    assert (await _access.spend_allowed(_Req()))[2] is None
    assert (await _access.spend_allowed(_Req()))[2] is None
    assert (await _access.spend_allowed(_Req()))[2] == "rate_limited"


async def test_side_calls_do_not_consume_the_agent_turn_allowance(monkeypatch):
    """
    The reason for two buckets. Each answered turn triggers one follow-ups call,
    so metering them together would have silently halved the documented turns
    per minute.
    """
    _wire(monkeypatch, RATE_LIMIT_TURNS_PER_MIN=2)
    for _ in range(2):
        await _access.spend_allowed(_Req())
    assert (await _access.spend_allowed(_Req()))[2] == "rate_limited"

    # The turn bucket is untouched.
    rate_limit.check_and_record("uuid-a")
    rate_limit.check_and_record("uuid-a")
    with pytest.raises(rate_limit.RateLimited):
        rate_limit.check_and_record("uuid-a")
    assert rate_limit.usage("uuid-a", bucket=rate_limit.AUX_BUCKET) == (2, 2)
