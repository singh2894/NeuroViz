"""Checks for the production-grade upgrades: leak-free one-hot pipelines,
scoring unseen data, time-aware CV, fairness, report export, and data IO."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from app.aie import evaluation, fairness, runner
from app.data_io import sample_sales, sheets_csv_url
from app.report import build_report


def _cat_frame(n: int = 120) -> pd.DataFrame:
    """Target is fully determined by a string category — the case the old
    factorize() encoding handled badly for linear models."""
    rng = np.random.default_rng(0)
    cats = rng.choice(["red", "green", "blue"], n)
    return pd.DataFrame(
        {
            "color": cats,
            "noise": rng.normal(size=n),
            "label": (cats == "red").astype(int),
        }
    )


def test_pipeline_one_hot_learns_categoricals():
    df = _cat_frame()
    X, y = runner.prepare_features(df, "label")
    model = runner.make_model(LogisticRegression(max_iter=1000), X)
    model.fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95


def test_pipeline_scores_unseen_category_and_nan():
    df = _cat_frame()
    X, y = runner.prepare_features(df, "label")
    model = runner.make_model(LogisticRegression(max_iter=1000), X)
    model.fit(X, y)
    new = pd.DataFrame({"color": ["purple", None, "red"], "noise": [0.0, np.nan, 1.0]})
    preds = model.predict(new[X.columns])
    assert len(preds) == 3  # unseen category + NaN must not crash


def test_prepare_features_epochizes_datetimes():
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-01-01", "2024-06-01", None]),
            "v": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )
    X, _ = runner.prepare_features(df, "target")
    assert pd.api.types.is_numeric_dtype(X["when"])  # epoch seconds, NaT -> NaN
    assert X["when"].isna().sum() == 1


def test_leaderboard_accepts_time_series_splitter():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame(
        {"x": np.arange(n, dtype=float), "y": np.arange(n) + rng.normal(size=n)}
    )
    X, y = runner.prepare_features(df, "y")
    lb = evaluation.build_leaderboard(
        [("lin", runner.make_model(LinearRegression(), X))],
        X,
        y,
        task="regression",
        cv_folds=3,
        splitter=TimeSeriesSplit(n_splits=3),
    )
    assert lb and "rmse" in lb[0].scores


def test_fairness_multiclass_and_alerts():
    y_true = pd.Series([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = pd.Series([0, 1, 2, 0, 0, 0, 0, 1])
    group = pd.Series(["a", "a", "a", "a", "b", "b", "b", "b"])
    gm = fairness.group_classification_metrics(y_true, y_pred, group)
    assert set(gm) == {"a", "b"}  # multiclass must not raise
    alerts = fairness.disparity_alerts(gm, metric="accuracy", max_gap=0.1)
    assert alerts  # group a is perfect, group b is not


def test_encode_target_keeps_index():
    y = pd.Series(["x", "y", "x"], index=[10, 20, 30])
    coded, classes = runner.encode_classification_target(y)
    assert list(coded.index) == [10, 20, 30]
    assert classes == ["x", "y"]


def test_sample_sales_is_deterministic_and_usable():
    a, b = sample_sales(), sample_sales()
    assert a.equals(b)
    assert a.height == 600
    assert {"date", "region", "product", "units", "revenue"} <= set(a.columns)


def test_sheets_url_rewrite():
    url = "https://docs.google.com/spreadsheets/d/abc1-XY/edit#gid=42"
    assert sheets_csv_url(url) == (
        "https://docs.google.com/spreadsheets/d/abc1-XY/export?format=csv&gid=42"
    )
    assert sheets_csv_url("https://example.com/data.csv") is None


def test_report_is_self_contained_html():
    pytest.importorskip("vl_convert")
    spec = {
        "data": {"values": [{"a": "A", "b": 3}, {"a": "B", "b": 7}]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "a", "type": "nominal"},
            "y": {"field": "b", "type": "quantitative"},
        },
    }
    html_out = build_report(
        title="Demo",
        kpis=[("Rows", "2")],
        charts=[(spec, "a by b")],
        stats_html="<table><tr><td>1</td></tr></table>",
    )
    assert html_out.startswith("<!doctype html>")
    assert "data:image/png;base64," in html_out  # chart embedded, no external refs
    assert "Rows" in html_out and "Demo" in html_out
