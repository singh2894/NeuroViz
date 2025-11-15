from app.parsers.nlp import parse_intent


def test_trend_parse():
    assert parse_intent("trend of sales over time").kind == "trend"
