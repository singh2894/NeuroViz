import pandas as pd

from app.aie import diagnostics as d


def test_missing_report_counts():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    report = d.missing_values_report(df)
    assert report.per_column.loc["b", "missing_count"] == 2
    assert report.per_column.loc["a", "missing_count"] == 1


def test_duplicate_report():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    dup = d.duplicate_report(df)
    assert dup.duplicate_count == 1
    assert abs(dup.duplicate_ratio - (1 / 3)) < 1e-6


def test_outlier_report_iqr():
    df = pd.DataFrame({"x": [1, 2, 3, 100]})
    out = d.outlier_report(df, numeric_cols=["x"], iqr_multiplier=1.5)
    assert out.per_column["x"]["outlier_count"] == 1


def test_correlation_hotspots():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": [1, 1, 1]})
    hot = d.correlation_hotspots(df, numeric_cols=["a", "b", "c"], threshold=0.8)
    pairs = [p[:2] for p in hot.pairs]
    assert ("a", "b") in pairs or ("b", "a") in pairs


def test_categorical_correlation_hotspots():
    df = pd.DataFrame(
        {
            "cat1": ["x", "x", "y", "y"],
            "cat2": ["m", "m", "n", "n"],
            "cat3": ["a", "b", "a", "b"],
        }
    )
    hot = d.categorical_correlation_hotspots(
        df, categorical_cols=["cat1", "cat2", "cat3"], threshold=0.5
    )
    pairs = [p[:2] for p in hot.pairs]
    assert ("cat1", "cat2") in pairs or ("cat2", "cat1") in pairs


def test_diagnose_aggregates():
    df = pd.DataFrame({"a": [1, None, 3], "b": [2, 4, 6]})
    summary = d.diagnose(df, numeric_cols=["b"])
    assert summary.duplicates.duplicate_count == 0
    assert "b" in summary.outliers.per_column
