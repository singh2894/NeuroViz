"""Tests for the intent parser (rule-based path and mocked LLM path)."""

import io
import json

import pytest

from app.parsers import nlp
from app.parsers.nlp import parse_intent


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Default to the rule-based path; LLM tests opt in explicitly."""
    monkeypatch.delenv("LLM_API_URL", raising=False)


def test_kind_detection():
    assert parse_intent("trend of sales over time").kind == "trend"
    assert parse_intent("compare revenue by region").kind == "compare"
    assert parse_intent("top products by profit").kind == "rank"
    assert parse_intent("distribution of customer age").kind == "distribution"


def test_kind_default_is_trend():
    assert parse_intent("something unrecognizable").kind == "trend"


def test_metric_synonyms():
    assert parse_intent("show revenue over time").metric == "sales"
    assert parse_intent("carbon emissions by country").metric == "co2"


def test_agg_and_time_grain():
    intent = parse_intent("average temperature monthly")
    assert intent.agg == "mean"
    assert intent.time_grain == "1mo"


def test_original_text_preserved():
    assert parse_intent("BMI vs age").text == "BMI vs age"


def test_fuzzy_column_matching():
    cols = ["BMI", "Age", "Weight (kg)"]
    intent = parse_intent("bmi vs age", columns=cols)
    assert intent.columns == ["BMI", "Age"]


def test_fuzzy_column_matching_tolerates_typos():
    intent = parse_intent("show me the wieght", columns=["weight", "height"])
    assert intent.columns == ["weight"]


def test_no_columns_given():
    assert parse_intent("trend of sales").columns == []


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_llm_reply(content: str):
    payload = {"choices": [{"message": {"content": content}}]}
    return _FakeResponse(json.dumps(payload).encode())


def test_llm_path_parses_reply(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
    reply = (
        '```json\n{"kind": "compare", "metric": "sales", "agg": "sum",'
        ' "time_grain": null, "columns": ["Region", "Sales"],'
        ' "filters": {"Region": "West", "Bogus": "x"}}\n```'
    )
    monkeypatch.setattr(
        nlp.urllib.request,
        "urlopen",
        lambda req, timeout: _fake_llm_reply(reply),
    )
    intent = parse_intent("total sales in the West region", columns=["Region", "Sales"])
    assert intent.kind == "compare"
    assert intent.agg == "sum"
    assert intent.columns == ["Region", "Sales"]
    assert intent.filters == {"Region": "West"}  # unknown columns dropped
    assert intent.source == "llm"
    assert intent.note is None


def test_llm_invalid_values_are_sanitized(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
    reply = '{"kind": "pie!!", "agg": "median", "columns": ["NotAColumn"]}'
    monkeypatch.setattr(
        nlp.urllib.request,
        "urlopen",
        lambda req, timeout: _fake_llm_reply(reply),
    )
    intent = parse_intent("trend of sales", columns=["Sales"])
    assert intent.kind == "trend"  # falls back to keyword detection
    assert intent.agg is None
    assert intent.columns == []


def test_answer_question_none_without_llm():
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})
    assert nlp.answer_question("what is the max of a", df) is None


def test_answer_question_with_mocked_llm(monkeypatch):
    import polars as pl

    monkeypatch.setenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
    monkeypatch.setattr(
        nlp.urllib.request,
        "urlopen",
        lambda req, timeout: _fake_llm_reply("The max of a is 3."),
    )
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert nlp.answer_question("what is the max of a", df) == "The max of a is 3."


def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")

    def boom(req, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(nlp.urllib.request, "urlopen", boom)
    intent = parse_intent("compare sales by region", columns=["region"])
    assert intent.kind == "compare"
    assert intent.columns == ["region"]
    assert intent.source == "rules"
    assert intent.note is not None  # failure is surfaced, not silent
    assert "offline parser" in intent.note
