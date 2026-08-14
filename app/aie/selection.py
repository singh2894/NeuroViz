"""
Layer 4 – Feature Selection.

Provides utilities to run filter/embedded/wrapper methods and combine feature importances.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (
    VarianceThreshold,
    chi2,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression


def _encode_non_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            X_enc[col] = pd.factorize(X_enc[col].astype(str))[0]
        elif X_enc[col].isna().any():
            # Ranking heuristics can't take NaN; median-fill (all-NaN -> 0).
            X_enc[col] = X_enc[col].fillna(X_enc[col].median()).fillna(0)
    return X_enc


def _infer_effective_task(task: str, y: pd.Series) -> (str, str):
    """
    If task is set to classification but the target looks continuous, fall back to regression scorers.
    Returns the effective task and an optional note.
    """
    note = ""
    if task == "classification":
        y_series = pd.Series(y)
        if pd.api.types.is_numeric_dtype(y_series):
            unique_vals = y_series.nunique(dropna=True)
            is_float = pd.api.types.is_float_dtype(y_series)
            if is_float or unique_vals > max(20, 0.1 * len(y_series)):
                task = "regression"
                note = "Target appears continuous; using regression scorers instead of classification."
    return task, note


def filter_methods(
    X: pd.DataFrame, y: pd.Series, task: str, variance_threshold: float = 0.0
) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}

    if variance_threshold > 0:
        vt = VarianceThreshold(threshold=variance_threshold)
        vt.fit(X)
        scores["variance"] = {
            col: float(vt.variances_[i]) for i, col in enumerate(X.columns)
        }

    X_enc = _encode_non_numeric(X)

    if task == "classification":
        try:
            mi = mutual_info_classif(X_enc, y, discrete_features="auto")
        except ValueError:
            # Fallback when the target isn't suitable for classification MI (e.g., too many unique values).
            mi = mutual_info_regression(X_enc, y, discrete_features="auto")
    else:
        mi = mutual_info_regression(X_enc, y, discrete_features="auto")
    scores["mutual_info"] = {col: float(mi[i]) for i, col in enumerate(X.columns)}

    # Chi-square only for non-negative features, mostly for categorical one-hot encoded
    if task == "classification":
        try:
            X_nonneg = X_enc.copy()
            X_nonneg[X_nonneg < 0] = 0
            chi_vals, _ = chi2(X_nonneg, y)
            scores["chi2"] = {
                col: float(chi_vals[i]) for i, col in enumerate(X.columns)
            }
        except ValueError:
            # If target isn't valid for chi-square, skip gracefully.
            scores["chi2"] = {}

    return scores


def embedded_methods(
    X: pd.DataFrame, y: pd.Series, task: str
) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    X = _encode_non_numeric(X)  # raw dtypes are welcome; idempotent if numeric
    if task == "classification":
        try:
            l1 = LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=2000, tol=1e-3
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                l1.fit(X, y)
            scores["l1"] = {
                col: float(abs(coef)) for col, coef in zip(X.columns, l1.coef_[0])
            }
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            scores["tree"] = {
                col: float(imp) for col, imp in zip(X.columns, rf.feature_importances_)
            }
            return scores
        except ValueError:
            # If target is not suitable for classification, fall back to regression scorers.
            task = "regression"

    # Regression path (used directly or as fallback).
    l1 = Lasso(alpha=0.001, max_iter=5000, tol=1e-3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        l1.fit(X, y)
    scores["l1"] = {col: float(abs(coef)) for col, coef in zip(X.columns, l1.coef_)}
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    scores["tree"] = {
        col: float(imp) for col, imp in zip(X.columns, rf.feature_importances_)
    }
    return scores


def unify_importances(score_dicts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, List[float]] = {}
    for method_scores in score_dicts.values():
        for feat, score in method_scores.items():
            agg.setdefault(feat, []).append(score)
    return {feat: float(np.mean(scores)) for feat, scores in agg.items()}
