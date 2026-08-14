import numpy as np
import pandas as pd

from app.aie import understanding as u


def test_infer_schema_and_convertible_text():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "date_col": pd.date_range("2020-01-01", periods=3, freq="D"),
            "text_num": ["1", "2", "x"],
        }
    )
    schema = u.infer_schema(df)
    assert schema.numeric == ["num"]
    assert schema.categorical == ["cat", "text_num"]
    assert schema.boolean == ["bool_col"]
    assert schema.datetime == ["date_col"]
    assert schema.convertible_text_numeric == ["text_num"]


def test_infer_target_and_task_classification_numeric_low_cardinality():
    df = pd.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    target_info = u.infer_target(df)
    assert target_info.target == "target"
    task_info = u.infer_task(
        df, target_info.target, classification_cardinality_cutoff=5
    )
    assert task_info.task == "classification"
    assert "low cardinality" in task_info.reason.lower()


def test_infer_task_regression_on_high_cardinality_numeric():
    df = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "target": [10, 20, 30, 40, 50]})
    task_info = u.infer_task(df, "target", classification_cardinality_cutoff=3)
    assert task_info.task == "regression"


def test_detect_imbalance_flags_small_minority():
    y = pd.Series([0] * 90 + [1] * 10)
    report = u.detect_imbalance(y, threshold=0.2)
    assert report.is_imbalanced is True
    assert report.ratio == 0.1
    assert report.class_counts[0] == 90 and report.class_counts[1] == 10


def test_detect_skew_identifies_skewed_feature():
    skewed = pd.Series(np.concatenate([np.ones(90), np.arange(10)]))
    df = pd.DataFrame({"skewed": skewed, "balanced": np.arange(100)})
    report = u.detect_skew(df, numeric_cols=["skewed", "balanced"], threshold=1.0)
    assert "skewed" in report.skewed_features
    assert "balanced" not in report.skewed_features


def test_detect_leakage_numeric_corr():
    df = pd.DataFrame(
        {
            "feature_leak": [1, 2, 3, 4, 5],
            "feature_noise": [5, 4, 3, 2, 1],
            "target": [2, 4, 6, 8, 10],
        }
    )
    leak_report = u.detect_leakage(
        df,
        target_col="target",
        numeric_cols=["feature_leak", "feature_noise", "target"],
        categorical_cols=[],
        corr_threshold=0.9,
    )
    assert "feature_leak" in leak_report.suspicious_features
    assert leak_report.method.startswith("abs(corr)")


def test_summarize_handles_missing_target():
    df = pd.DataFrame({"id": [1, 2, 3], "feature": [0, 1, 0]})
    summary = u.summarize(df, target_hint=None)
    assert summary.target.target is None
    assert summary.task.task is None
    assert summary.skew.skewed_features == {}
