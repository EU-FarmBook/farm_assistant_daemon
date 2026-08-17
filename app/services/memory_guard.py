# app/services/memory_guard.py
"""
What may become a memory, and what may not.

The failures this exists to stop, both observed in the pilot: a user in the
Netherlands asked what pig manure is used for in Italy, and the agent stored
"farms in Italy"; the same user asked something once in another language, and it
stored "prefers responses in Hungarian" — which then fought the per-turn
language rule on every later answer. Asking about a place, a crop or a practice says nothing about
the person asking — but to a model mid-conversation the two look alike, and once
written the mistake is durable, shapes every later answer, and the user has to
find and delete it.

So a note is not written because the agent wants to write it. It is written only
if the USER'S OWN MESSAGE in that turn supports it. The check is made against the
message text, by a model asked a single narrow question — not by a word list,
which would be brittle and English-only, and this platform serves 24 languages.

Three tiers, and only the middle one goes through here:

  USER.md   `about_you` + custom instructions. The user writes it. Authoritative;
            the agent may never contradict or overwrite it.
  MEMORY.md agent-written notes. Provisional, grounded in the user's own words,
            and subordinate to USER.md wherever the two disagree.
  transcript per-session chat history. Never promoted to either.
"""

import logging
from typing import Optional, Tuple

import httpx

from app.config import get_settings

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.memory-guard")

# Cheap structural rejections, before any model call. None of these judge
# meaning — they only catch shapes that cannot be a durable fact about a person.
_MAX_FACT_CHARS = 200
_MIN_FACT_CHARS = 8

_VERDICT_PROMPT = """A conversational assistant wants to store a durable fact about its user.

The user's message this turn was:
---
{message}
---

The fact it wants to store is:
---
{fact}
---

Answer YES if the user's message ASSERTS that fact about THEMSELVES — their own
farm, work, location, crops, livestock, role, or a preference they stated.

Judge the SUBSTANCE, not the wording. "I live in the Netherlands" supports "the
user is based in the Netherlands" and "the user farms in the Netherlands" when
the conversation is about their farm. A CORRECTION is always YES: "I'm in the
Netherlands, not France" asserts the Netherlands. Users must be able to fix what
you know about them by saying so.

Answer NO if:
- the fact is merely the TOPIC of a question ("what is pig manure used for in
  Italy?" does not mean the user is in Italy),
- it is about someone or something else, not the user,
- it is speculation, inference, or a detail the assistant supplied rather than
  the user,
- it restates the assistant's own answer,
- it is transient (what they are doing right now, this one question),
- it is about the LANGUAGE the message happens to be written in. Asking a
  question in Hungarian does not make someone Hungarian or mean they want
  Hungarian answers. Only YES if they explicitly asked to always be answered in
  a given language,
- it is sensitive: health, finances, political or religious views, or anything
  about a third party.

Reply with exactly one word: YES or NO.
"""


def structural_reject(fact: str) -> Optional[str]:
    """Reasons to refuse without spending a model call. Returns a reason or None."""
    text = (fact or "").strip()
    if len(text) < _MIN_FACT_CHARS:
        return "too short to be a durable fact"
    if len(text) > _MAX_FACT_CHARS:
        return "too long to be a single fact"
    if "?" in text:
        # A stored question is a topic, and topics are what caused this module.
        return "a question is a topic, not a fact about the user"
    return None


async def is_supported_by_user(fact: str, user_message: str) -> Tuple[bool, str]:
    """
    Does the user's own message support this fact about them?

    Returns (allowed, reason). Fails CLOSED: if the check cannot run, the note is
    not written. An unwritten true fact costs the user one repetition; a written
    false one shapes every future answer until they find and delete it, which is
    the asymmetry this whole module is about.
    """
    reason = structural_reject(fact)
    if reason:
        return False, reason

    if not (user_message or "").strip():
        return False, "no user message to check the fact against"

    if not S.MISTRAL_API_KEY:
        return False, "no provider key available to validate the fact"

    prompt = _VERDICT_PROMPT.format(message=user_message.strip()[:2000], fact=fact.strip())

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=3.0),
            verify=S.VERIFY_SSL,
        ) as client:
            r = await client.post(
                f"{S.MISTRAL_API_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {S.MISTRAL_API_KEY}"},
                json={
                    "model": S.MEMORY_SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    # A judgement, not a composition: no room for creativity.
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
            )
        if r.status_code != 200:
            logger.warning("Memory validation returned HTTP %s", r.status_code)
            return False, "validation unavailable"

        choices = (r.json() or {}).get("choices") or []
        verdict = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
    except (httpx.HTTPError, ValueError, IndexError, AttributeError) as e:
        logger.warning("Memory validation failed: %s", e)
        return False, "validation unavailable"

    if verdict.strip().upper().startswith("YES"):
        return True, "supported by the user's own words"
    return False, "not asserted by the user about themselves"
