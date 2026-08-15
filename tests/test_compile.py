"""Tests for the chart compiler across all four intent kinds."""

from datetime import date

import polars as pl

from app.compilers.altair_compile import compile as compile_chart
from app.compilers.altair_compile import infer_schema
from app.parsers.nlp import Intent


def test_infer_schema_sees_all_numeric_widths_and_booleans():
    df = pl.DataFrame(
        {
            "u32": pl.Series([1, 2], dtype=pl.UInt32),
            "i16": pl.Series([3, 4], dtype=pl.Int16),
            "f32": pl.Series([1.0, 2.0], dtype=pl.Float32),
            "flag": [True, False],
            "name": ["a", "b"],
        }
    )
    schema = infer_schema(df)
    assert set(schema["numeric_cols"]) == {"u32", "i16", "f32"}
    assert set(schema["categorical_cols"]) == {"flag", "name"}


def _dated_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.date_range(
                date(2021, 1, 1), date(2021, 1, 10), "1d", eager=True
            ),
            "sales": [float(i) for i in range(10)],
            "region": ["north", "south"] * 5,
        }
    )


def test_trend_chart():
    intent = Intent(kind="trend", metric="sales", text="trend of sales over time")
    chart, caption = compile_chart(intent, _dated_df())
    assert chart is not None
    assert "sales" in caption


def test_distribution_is_prebinned_server_side():
    df = pl.DataFrame({"value": [float(i) for i in range(100_000)]})
    intent = Intent(kind="distribution", text="distribution of value")
    chart, caption = compile_chart(intent, df)
    assert chart is not None
    assert len(chart.data) <= 30  # 30 bars reach the browser, never 100k rows


def test_scatter_caps_points_sent_to_browser():
    n = 25_000
    df = pl.DataFrame(
        {"aaa": [float(i) for i in range(n)], "bbb": [float(i) for i in range(n)]}
    )
    intent = Intent(kind="compare", text="aaa vs bbb")
    chart, caption = compile_chart(intent, df)
    assert chart is not None
    assert len(chart.data) <= 20_000
    assert "sample" in caption


def test_trend_auto_aggregates_large_raw_series():
    dates = pl.date_range(date(2021, 1, 1), date(2021, 1, 30), "1d", eager=True)
    df = pl.DataFrame(
        {
            "date": pl.concat([dates] * 2_000),
            "sales": [1.0] * (30 * 2_000),
        }
    )
    intent = Intent(kind="trend", text="trend of sales")
    chart, caption = compile_chart(intent, df)
    assert chart is not None
    assert len(chart.data) <= 31  # 60k raw rows collapse to one point per date
    assert "auto-aggregated" in caption


def test_trend_datetime_axis_buckets_to_days():
    from datetime import datetime, timedelta

    start = datetime(2021, 1, 1)
    stamps = [start + timedelta(minutes=i) for i in range(60_000)]  # ~42 days
    df = pl.DataFrame({"pickup_time": stamps, "fare": [1.0] * 60_000})
    intent = Intent(kind="trend", text="trend of fares")
    chart, caption = compile_chart(intent, df)
    assert chart is not None
    assert len(chart.data) <= 45  # one point per day, never per timestamp


def test_trend_time_grain_buckets_dates():
    df = pl.DataFrame(
        {
            "date": pl.date_range(date(2021, 1, 1), date(2021, 3, 1), "1d", eager=True),
            "sales": [1.0] * 60,
        }
    )
    intent = Intent(kind="trend", time_grain="1mo", text="monthly trend of sales")
    chart, caption = compile_chart(intent, df)
    assert chart is not None
    assert len(chart.data) == 3  # Jan, Feb, Mar buckets
    assert "per month" in caption


def test_compare_chart():
    intent = Intent(kind="compare", metric="sales", text="compare sales by region")
    chart, caption = compile_chart(intent, _dated_df())
    assert chart is not None


def test_rank_uses_category_chart():
    intent = Intent(kind="rank", metric="sales", text="top regions by sales")
    chart, caption = compile_chart(intent, _dated_df())
    assert chart is not None
    assert "region" in caption


def test_distribution_chart():
    intent = Intent(kind="distribution", metric="sales", text="distribution of sales")
    chart, caption = compile_chart(intent, _dated_df())
    assert chart is not None
    assert "Distribution" in caption


def test_scatter_from_two_numeric_mentions():
    df = pl.DataFrame({"bmi": [20.0, 25.0, 30.0], "age": [30.0, 40.0, 50.0]})
    intent = Intent(kind="compare", text="bmi vs age")
    chart, caption = compile_chart(intent, df)
    assert chart is not None
    assert "Scatter" in caption


def test_llm_columns_take_priority():
    df = pl.DataFrame(
        {"bmi": [20.0, 25.0], "age": [30.0, 40.0], "weight": [60.0, 70.0]}
    )
    intent = Intent(
        kind="compare", columns=["weight", "age"], text="weight against age"
    )
    chart, caption = compile_chart(intent, df)
    assert chart is not None
    assert "weight" in caption and "age" in caption


def test_no_chart_message_when_impossible():
    df = pl.DataFrame({"name": ["a", "b"]})
    intent = Intent(kind="distribution", text="distribution of anything")
    chart, caption = compile_chart(intent, df)
    assert chart is None
    assert caption  # a human-readable reason is returned
