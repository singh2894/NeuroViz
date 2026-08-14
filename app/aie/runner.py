"""
Orchestrator for quick end-to-end run: model selection + CV leaderboard.
This is a minimal bridge for the UI until full tuning/training wiring is added.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def encode_classification_target(y: pd.Series) -> Tuple[pd.Series, List[str]]:
    """
    Factorize string/object labels into integer codes for model compatibility (e.g., XGBoost).
    Returns the encoded series and the list of original class labels by code index.
    """
    codes, uniques = pd.factorize(pd.Series(y).astype(str), sort=True)
    # Keep the original index so downstream splits stay aligned with X.
    return pd.Series(codes, name=y.name, index=y.index), list(uniques)


def _drop_identifier_columns(
    df: pd.DataFrame, min_unique_ratio: float = 0.9
) -> pd.DataFrame:
    """
    Remove obvious identifier/name-like columns and near-unique columns that don't help prediction.
    """
    to_drop: List[str] = []
    id_tokens = ["id", "identifier", "uuid", "guid", "serial", "name", "employee"]
    for col in df.columns:
        norm = str(col).strip().lower()
        if any(tok in norm for tok in id_tokens):
            to_drop.append(col)
            continue
        # Near-unique only signals an identifier for non-numeric columns —
        # continuous numeric features are naturally unique and must be kept.
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        nunique = df[col].nunique(dropna=True)
        if len(df) and nunique >= min_unique_ratio * len(df):
            to_drop.append(col)
    # Never drop every feature: an empty X breaks every estimator downstream.
    if len(to_drop) == df.shape[1]:
        return df
    return df.drop(columns=to_drop) if to_drop else df


def epochize_datetimes(X: pd.DataFrame) -> pd.DataFrame:
    """Datetimes become epoch seconds; anything unconvertible is dropped.
    Used at feature-prep AND predict time so both see identical columns."""
    X = X.copy()
    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            try:
                tz = getattr(X[col].dt, "tz", None)
                X[col] = (X[col] - pd.Timestamp(0, tz=tz)).dt.total_seconds()
            except (TypeError, ValueError):
                X = X.drop(columns=[col])
    return X


def prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split X/y keeping raw dtypes: encoding/imputation live inside the
    model pipeline (build_preprocessor) so they are fit per CV fold —
    leak-free — and new raw data can be scored directly."""
    y = df[target]
    X = df.drop(columns=[target])
    X = _drop_identifier_columns(X)
    return epochize_datetimes(X), y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute numerics; mode-impute + one-hot categoricals (capped at
    20 levels; unseen values at predict time fold into the infrequent bucket).

    Replaces the old factorize() baseline, which fed fake ordinality to
    linear/KNN/SVM models."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=20,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_model(estimator, X: pd.DataFrame) -> Pipeline:
    """Wrap an estimator with the preprocessor: every fit/predict runs the
    same encoding, so CV stays leak-free and raw rows score directly."""
    return Pipeline([("prep", build_preprocessor(X)), ("est", clone(estimator))])
