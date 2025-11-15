import polars as pl

from app.compilers.altair_compile import compile as compile_chart
from app.parsers.nlp import parse_intent


def test_compile_trend_minimal():
    df = pl.DataFrame(
        {
            "date": [
                pl.date_range(
                    low=pl.date(2021, 1, 1), high=pl.date(2021, 1, 10), interval="1d"
                )
            ],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    intent = parse_intent("trend over time")
    chart, cap = compile_chart(intent, df)
    assert chart is not None
