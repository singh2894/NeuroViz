"""
Layer 3 – Auto-Cleaning Engine.

Implements cleaning primitives as sklearn-style transformers so they can
participate in downstream pipelines. Also produces a cleaning summary for UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


@dataclass
class CleaningSummary:
    steps: List[str]
    imputation: Dict[str, str]
    dropped_duplicates: int
    dropped_constant: List[str]
    dropped_sparse: List[str]
    outlier_capped: List[str]
    notes: List[str]


class NumericImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strategy: str = "median",
        knn_neighbors: int = 5,
        columns: Optional[List[str]] = None,
    ):
        self.strategy = strategy
        self.knn_neighbors = knn_neighbors
        self.columns = columns
        self.fill_values_: Dict[str, float] = {}
        self.knn_imputer_: Optional[KNNImputer] = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        cols = self.columns or df.columns.tolist()
        if self.strategy == "median":
            for c in cols:
                self.fill_values_[c] = df[c].median()
        elif self.strategy == "mean":
            for c in cols:
                self.fill_values_[c] = df[c].mean()
        elif self.strategy == "knn":
            self.knn_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
            self.knn_imputer_.fit(df[cols])
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        cols = self.columns or df.columns.tolist()
        if self.strategy in {"median", "mean"}:
            for c in cols:
                df[c] = df[c].fillna(self.fill_values_[c])
        else:
            df[cols] = self.knn_imputer_.transform(df[cols])
        return df


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self, fill_value: str = "Unknown", columns: Optional[List[str]] = None
    ):
        self.fill_value = fill_value
        self.columns = columns
        self.modes_: Dict[str, str] = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        cols = self.columns or df.columns.tolist()
        for c in cols:
            # For simplicity and robustness, always use fill_value to avoid propagating noisy modes.
            self.modes_[c] = self.fill_value
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        cols = self.columns or df.columns.tolist()
        for c in cols:
            fill_val = self.modes_.get(c, self.fill_value)
            df[c] = df[c].fillna(fill_val)
        return df


class MissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        new_cols = {
            f"{c}_missing": df[c].isna().astype(int)
            for c in self.columns
            if c in df.columns
        }
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df


class TypeCoercer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        to_numeric: Optional[List[str]] = None,
        to_datetime: Optional[List[str]] = None,
        to_bool: Optional[List[str]] = None,
    ):
        self.to_numeric = to_numeric or []
        self.to_datetime = to_datetime or []
        self.to_bool = to_bool or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for c in self.to_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in self.to_datetime:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in self.to_bool:
            df[c] = (
                df[c]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
        return df


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str], iqr_multiplier: float = 1.5):
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier
        self.bounds_: Dict[str, tuple] = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        for c in self.columns:
            series = df[c].dropna()
            if series.empty:
                continue
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self.bounds_[c] = (lower, upper)
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for c, (lower, upper) in self.bounds_.items():
            df[c] = df[c].clip(lower=lower, upper=upper)
        return df


def drop_duplicates(df: pd.DataFrame) -> (pd.DataFrame, int):
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)


def drop_constant_and_sparse(
    df: pd.DataFrame,
    sparse_threshold: float = 0.95,
) -> (pd.DataFrame, List[str], List[str]):
    constant = []
    sparse = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=False)
        if nunique <= 1:
            constant.append(c)
            continue
        most_freq = df[c].value_counts(normalize=True, dropna=False).iloc[0]
        if most_freq >= sparse_threshold:
            sparse.append(c)
    df = df.drop(columns=constant + sparse)
    return df, constant, sparse


def clean_dataset(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    type_coercions: Optional[Dict[str, List[str]]] = None,
    missing_indicator: bool = True,
    numeric_impute_strategy: str = "median",
    cat_fill_value: str = "Unknown",
    iqr_multiplier: float = 1.5,
    sparse_threshold: float = 0.95,
) -> (pd.DataFrame, CleaningSummary):
    steps = []
    notes = []

    # Normalize column names minimally to avoid leading/trailing whitespace drift.
    normalize = lambda c: str(c).strip()
    df = df.rename(columns=normalize)
    numeric_cols = [normalize(c) for c in numeric_cols]
    categorical_cols = [normalize(c) for c in categorical_cols]

    # Guard against schema/column drift between understanding and cleaning.
    missing_numeric = [c for c in numeric_cols if c not in df.columns]
    missing_categorical = [c for c in categorical_cols if c not in df.columns]
    if missing_numeric or missing_categorical:
        notes.append(
            f"Columns missing at cleaning time; skipped numeric={missing_numeric or '[]'}, categorical={missing_categorical or '[]'}"
        )
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Type coercion
    coercer = TypeCoercer(
        to_numeric=type_coercions.get("to_numeric", []) if type_coercions else [],
        to_datetime=type_coercions.get("to_datetime", []) if type_coercions else [],
        to_bool=type_coercions.get("to_bool", []) if type_coercions else [],
    )
    df = coercer.transform(df)
    steps.append("type_coercion")

    # Missingness indicators
    if missing_indicator:
        indicator = MissingIndicator(columns=numeric_cols + categorical_cols)
        df = indicator.transform(df)
        steps.append("missing_indicator_added")

    # Imputation
    num_imputer = NumericImputer(strategy=numeric_impute_strategy, columns=numeric_cols)
    num_imputer.fit(df[numeric_cols])
    df[numeric_cols] = num_imputer.transform(df[numeric_cols])

    cat_imputer = CategoricalImputer(
        fill_value=cat_fill_value, columns=categorical_cols
    )
    cat_imputer.fit(df[categorical_cols])
    df[categorical_cols] = cat_imputer.transform(df[categorical_cols])
    steps.append("imputation")

    # Outlier capping
    capper = OutlierCapper(columns=numeric_cols, iqr_multiplier=iqr_multiplier)
    capper.fit(df[numeric_cols])
    df[numeric_cols] = capper.transform(df[numeric_cols])
    steps.append("outlier_cap")

    # Drop duplicates
    df, dup_count = drop_duplicates(df)
    if dup_count:
        steps.append("drop_duplicates")

    # Drop constant / sparse
    df, constant, sparse = drop_constant_and_sparse(
        df, sparse_threshold=sparse_threshold
    )
    if constant:
        steps.append("drop_constant")
    if sparse:
        steps.append("drop_sparse")

    summary = CleaningSummary(
        steps=steps,
        imputation={
            "numeric": numeric_impute_strategy,
            "categorical": f"mode/{cat_fill_value}",
        },
        dropped_duplicates=dup_count,
        dropped_constant=constant,
        dropped_sparse=sparse,
        outlier_capped=list(capper.bounds_.keys()),
        notes=notes,
    )
    return df, summary
