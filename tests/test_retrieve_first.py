"""
Retrieve-first behaviour.

v2 cannot answer ungrounded because it retrieves before generating. A pure agent
can, and did: it told a user EU-FarmBook had nothing on pig manure without
running a single search. These tests pin the floor that fixes it — and the
property that makes v3 better rather than merely equal, which is that the agent
can still search again on top.
"""

import inspect

from app.routers import ask


def _source() -> str:
    return inspect.getsource(ask.stream_message)


def test_a_substantive_turn_retrieves_before_the_model_runs():
    src = _source()
    # Pre-retrieval must happen before stream_chat, not as a fallback after it.
    assert src.index("search_eu_farmbook") < src.index("stream_chat")


def test_prefetch_uses_the_same_tool_as_the_agent():
    # Same function => one citation register => a later agent hop continues the
    # numbering instead of restarting at [1] against different documents.
    assert "tool_server.search_eu_farmbook(" in _source()


def test_passages_are_handed_over_with_their_numbers():
    src = _source()
    assert "[{p['n']}] {p['text']}" in src
    assert "Cite what you use by these numbers" in src


def test_weak_results_ask_the_agent_to_search_again():
    src = _source()
    # v2 silently DROPS weak context and answers anyway; v3 tells the agent it
    # was weak and lets it re-query. That difference is the point.
    assert 'quality == "weak"' in src
    assert "search again" in src.lower()


def test_empty_retrieval_does_not_authorise_a_no_material_claim():
    src = _source()
    assert "returned nothing" in src
    assert "before concluding" in src


def test_very_short_messages_skip_the_search():
    src = _source()
    # A greeting should not cost an OpenSearch round trip. Length-based, never a
    # keyword list: those are brittle and English-only.
    assert "_PREFETCH_MIN_CHARS" in src
    assert ask._PREFETCH_MIN_CHARS > 0


def test_short_follow_ups_are_searched_with_their_context():
    history = [
        {"role": "user", "content": "What is pig manure used for?"},
        {"role": "assistant", "content": "It is used as an organic fertiliser."},
    ]
    # "and for maize?" retrieves nothing useful on its own.
    assert ask._search_query("and for maize?", history) == (
        "What is pig manure used for? and for maize?"
    )


def test_a_self_contained_question_is_searched_as_written():
    history = [{"role": "user", "content": "What is pig manure used for?"}]
    long_q = "Which cover crops suit a wet Atlantic climate on heavy clay soils in Brittany?"
    assert ask._search_query(long_q, history) == long_q


def test_search_query_without_history_is_the_question():
    assert ask._search_query("What is pig manure used for?", []) == (
        "What is pig manure used for?"
    )


def test_uncited_answers_do_not_claim_grounding():
    src = _source()
    # "Who am I?" pre-retrieved five unrelated documents and the UI said
    # "Grounded in EU-FarmBook" over an answer that cited none of them.
    assert 'sent_version and not re.search' in src
    assert 'cited nothing' in src


def test_the_agent_can_forget_a_wrong_note():
    from app.services import tool_server
    # Correcting a memory by adding a contradicting note leaves the user with
    # both, which is what produced "Italy (though I also have a note about
    # Provence-Alpes-Côte d'Azur)".
    assert hasattr(tool_server, "forget_about_user")
