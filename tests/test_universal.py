"""Universal-compatibility fuzz: every pathological dataset shape must pass
through the whole backend without raising."""

from datetime import date

import polars as pl
import pytest

from app.aie import cleaning, diagnostics, evaluation, models, runner, understanding
from app.compilers.altair_compile import compile as compile_chart
from app.components.filters import apply_filters
from app.parsers.nlp import Intent


def _nasty_datasets() -> dict[str, pl.DataFrame]:
    return {
        "empty_rows": pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)}),
        "one_row": pl.DataFrame({"x": [1.0], "y": ["only"]}),
        "one_column_numeric": pl.DataFrame({"just_numbers": [1.0, 2.0, 3.0]}),
        "one_column_text": pl.DataFrame({"just_text": ["a", "b", "c"]}),
        "all_null": pl.DataFrame(
            {"n": pl.Series([None, None], dtype=pl.Float64), "s": ["x", "y"]}
        ),
        "all_text": pl.DataFrame(
            {"name": ["ann", "bob"], "city": ["oslo", "rome"], "note": ["?", "-"]}
        ),
        "all_numeric": pl.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}
        ),
        "weird_names": pl.DataFrame(
            {
                "col with spaces": [1.0, 2.0],
                "Ünïcodé 🎉": ["a", "b"],
                "trailing ": [3.0, 4.0],
                "row": ["p", "q"],  # collides with heatmap internals
                "value": [5.0, 6.0],  # collides with melt internals
                "col": ["r", "s"],
            }
        ),
        "constant": pl.DataFrame({"k": [7, 7, 7], "s": ["same", "same", "same"]}),
        "high_cardinality": pl.DataFrame(
            {"id_text": [f"row-{i}" for i in range(300)], "v": list(range(300))}
        ),
        "inf_and_nan": pl.DataFrame(
            {
                "x": [1.0, float("inf"), float("-inf"), float("nan")],
                "g": ["a", "b", "a", "b"],
            }
        ),
        "bool_and_dates": pl.DataFrame(
            {
                "flag": [True, False, True],
                "when": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
                "amount": [1.0, 2.0, 3.0],
            }
        ),
        "year_column": pl.DataFrame(
            {"Year": [2019, 2020, 2021], "metric": [5.0, 6.0, 7.0]}
        ),
        "mixed_junk": pl.DataFrame(
            {
                "maybe_num": ["1", "2", "N/A", "?"],
                "maybe_date": ["2020-01-01", "not a date", "2021-02-02", ""],
                "target": ["yes", "no", "yes", "no"],
            }
        ),
    }


NASTY = _nasty_datasets()
QUERIES = [
    "trend over time",
    "compare by category",
    "top values",
    "distribution",
]


def _sanitized_pandas(df: pl.DataFrame):
    import numpy as np

    return df.to_pandas().replace([np.inf, -np.inf], np.nan)


@pytest.mark.parametrize("name", list(NASTY))
def test_charts_never_raise(name):
    df = NASTY[name]
    for q in QUERIES:
        for kind in ("trend", "compare", "rank", "distribution"):
            intent = Intent(kind=kind, text=q)
            chart, caption = compile_chart(intent, df)
            assert caption  # chart may be None, but a reason must come back


@pytest.mark.parametrize("name", list(NASTY))
def test_filters_never_raise(name):
    df = NASTY[name]
    # numeric range + categorical membership on arbitrary columns
    for col, dtype in zip(df.columns, df.dtypes, strict=False):
        if dtype.is_numeric():
            apply_filters(df, {col: (0.0, 10.0)})
        else:
            apply_filters(df, {col: ["a", "x"]})


@pytest.mark.parametrize("name", list(NASTY))
def test_ml_understanding_and_diagnosis_never_raise(name):
    pdf = _sanitized_pandas(NASTY[name])
    summary = understanding.summarize(pdf)
    diagnostics.diagnose(
        pdf,
        numeric_cols=summary.schema.numeric,
        categorical_cols=[
            c for c in summary.schema.categorical if pdf[c].nunique() <= 100
        ],
    )


@pytest.mark.parametrize("name", list(NASTY))
def test_ml_cleaning_never_raises(name):
    pdf = _sanitized_pandas(NASTY[name])
    summary = understanding.summarize(pdf)
    convertible = [
        c
        for c in getattr(summary.schema, "convertible_text_numeric", [])
        if c in pdf.columns
    ]
    cleaning.clean_dataset(
        pdf,
        numeric_cols=[
            c for c in list(summary.schema.numeric) + convertible if c in pdf.columns
        ],
        categorical_cols=[
            c
            for c in summary.schema.categorical
            if c in pdf.columns and c not in convertible
        ],
        type_coercions={"to_numeric": convertible},
    )


@pytest.mark.parametrize("name", ["all_numeric", "bool_and_dates", "mixed_junk"])
def test_ml_leaderboard_never_raises(name):
    pdf = _sanitized_pandas(NASTY[name])
    target = pdf.columns[-1]
    work = pdf.dropna(subset=[target])
    if work.empty:
        return
    X, y = runner.prepare_features(work, target)
    if X.empty:
        return
    task, _ = evaluation.infer_effective_task("classification", y)
    if task == "classification":
        y, _ = runner.encode_classification_target(y)
    candidates = models.select_models(
        task=task, n_rows=len(work), n_features=X.shape[1], n_categorical=1
    )
    evaluation.build_leaderboard(
        [(s.name, runner.make_model(s.estimator, X)) for s in candidates],
        X,
        y,
        task=task,
        cv_folds=2,
    )
