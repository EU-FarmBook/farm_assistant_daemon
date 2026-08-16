"""
Tests for the retrieval-quality verdict handed back to the agent.

v2 uses these same numbers to silently drop weak context. v3 reports them
instead, because the premise of the agent route is that the model decides what
to do about weak retrieval — and it cannot decide anything useful from an
unlabelled list of passages.
"""

import pytest

from app.config import Settings
from app.services import tool_server


def _item(title: str, description: str = "", semantic: float | None = None) -> dict:
    src = {"title": title, "description": description}
    item = {"_id": title, "_score": 5.0, "_source": src}
    if semantic is not None:
        item["semantic_score"] = semantic
        src["semantic_score"] = semantic
    return item


@pytest.fixture
def overlap_mode(monkeypatch):
    monkeypatch.setattr(tool_server, "S", Settings(RELEVANCE_MODE="overlap", _env_file=None))


@pytest.fixture
def semantic_mode(monkeypatch):
    monkeypatch.setattr(tool_server, "S", Settings(RELEVANCE_MODE="semantic", _env_file=None))


def test_on_topic_results_are_strong(overlap_mode):
    q = "soil nitrogen management for winter wheat"
    items = [
        _item("Soil nitrogen management in winter wheat",
              "Nitrogen fertilisation strategies for winter wheat on European soils"),
        _item("Winter wheat nitrogen trials", "Soil nitrogen uptake and management"),
    ]
    verdict = tool_server._assess(q, items)
    assert verdict["mode"] == "overlap"
    assert verdict["verdict"] == "strong"
    assert 0.0 <= verdict["score"] <= 1.0


def test_off_topic_results_are_weak_but_not_dropped(overlap_mode):
    q = "soil nitrogen management for winter wheat"
    items = [_item("Beekeeping equipment catalogue", "Hive tools and protective clothing")]
    verdict = tool_server._assess(q, items)
    assert verdict["verdict"] == "weak"
    # The point of the design: v3 reports and keeps going. Nothing here filters
    # the items out — that decision belongs to the agent.
    assert verdict["score"] < verdict["threshold"]


def test_semantic_mode_uses_the_semantic_threshold(semantic_mode):
    items = [_item("Anything", "Anything", semantic=0.95)]
    verdict = tool_server._assess("some query", items)
    assert verdict["mode"] == "semantic"
    assert verdict["threshold"] == pytest.approx(0.88)
    assert verdict["verdict"] == "strong"


def test_semantic_mode_falls_back_to_overlap_without_scores(semantic_mode):
    # scout only attaches semantic_score when asked; if it is missing the helper
    # returns None and we must not report a semantic verdict off a null.
    items = [_item("Soil nitrogen", "Nitrogen in soil")]
    verdict = tool_server._assess("soil nitrogen", items)
    assert verdict["mode"] == "overlap"
