import pandas as pd

from app.aie import cleaning as c


def test_numeric_imputer_median():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    imputer = c.NumericImputer(strategy="median", columns=["x"]).fit(df)
    transformed = imputer.transform(df)
    assert transformed["x"].isna().sum() == 0
    assert transformed["x"].iloc[1] == 2.0


def test_categorical_imputer_unknown():
    df = pd.DataFrame({"cat": ["a", None, "b"]})
    imputer = c.CategoricalImputer(columns=["cat"]).fit(df)
    transformed = imputer.transform(df)
    assert transformed["cat"].iloc[1] == "Unknown"


def test_missing_indicator_added():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "y"]})
    ind = c.MissingIndicator(columns=["a", "b"])
    transformed = ind.transform(df)
    assert "a_missing" in transformed.columns and "b_missing" in transformed.columns
    assert transformed["a_missing"].sum() == 1
    assert transformed["b_missing"].sum() == 1


def test_outlier_capper_clips_values():
    df = pd.DataFrame({"x": [1, 2, 3, 100]})
    capper = c.OutlierCapper(columns=["x"], iqr_multiplier=1.5).fit(df)
    transformed = capper.transform(df)
    assert transformed["x"].max() < 100


def test_clean_dataset_pipeline():
    df = pd.DataFrame(
        {
            "num": [1, None, 3, 100],
            "cat": ["a", None, "b", "b"],
        }
    )
    cleaned, summary = c.clean_dataset(
        df,
        numeric_cols=["num"],
        categorical_cols=["cat"],
        missing_indicator=True,
        numeric_impute_strategy="median",
        cat_fill_value="Unknown",
        iqr_multiplier=1.5,
        sparse_threshold=0.95,
    )
    assert cleaned["num"].isna().sum() == 0
    assert cleaned["cat"].isna().sum() == 0
    assert "num_missing" in cleaned.columns
    assert "drop_duplicates" not in summary.steps  # none dropped


def test_cleaning_summary_has_fields_the_clean_page_reads():
    """Regression guard: app/pages_ml.py reads exactly these attributes."""
    df = pd.DataFrame({"num": [1.0, 2.0, 2.0], "cat": ["a", "b", "b"]})
    _, summary = c.clean_dataset(df, numeric_cols=["num"], categorical_cols=["cat"])
    for field in (
        "steps",
        "imputation",
        "dropped_duplicates",
        "dropped_constant",
        "dropped_sparse",
        "outlier_capped",
        "notes",
    ):
        assert hasattr(summary, field), field
