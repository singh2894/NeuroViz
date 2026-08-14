"""
Layer 7 – Evaluation and Leaderboard.

Provides:
- Cross-validated scoring for classification/regression.
- Metric bundles per task.
- Leaderboard assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class LeaderboardEntry:
    model_name: str
    scores: Dict[str, float]
    fit_time: float


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    # Use weighted averages for multiclass to avoid crashes on non-binary targets.
    classes = pd.Series(y_true).dropna().unique()
    avg = "binary" if len(classes) <= 2 else "weighted"
    scores = {
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            # If probabilities are 2D, assume column 1 is the positive class.
            prob_vec = (
                y_proba[:, 1]
                if hasattr(y_proba, "ndim") and y_proba.ndim > 1
                else y_proba
            )
            scores["roc_auc"] = roc_auc_score(y_true, prob_vec)
            scores["pr_auc"] = average_precision_score(y_true, prob_vec)
        except Exception:
            # If prob-based metrics fail (e.g., multiclass without probability), skip gracefully.
            pass
    return scores


def infer_effective_task(task: str, y) -> Tuple[str, str]:
    """
    If task is declared classification but the target appears continuous/high-cardinality,
    fall back to regression metrics/splitting.
    """
    note = ""
    if task == "classification":
        y_series = pd.Series(y)
        if pd.api.types.is_numeric_dtype(y_series):
            unique_vals = y_series.nunique(dropna=True)
            is_float = pd.api.types.is_float_dtype(y_series)
            if is_float or unique_vals > max(20, 0.1 * len(y_series)):
                task = "regression"
    return task, note


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None)))
        * 100,
    }


def _aggregate_metrics(per_fold: List[Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    keys = set().union(*per_fold) if per_fold else set()
    for k in keys:
        vals = [m[k] for m in per_fold if k in m and np.isfinite(m[k])]
        if not vals:
            continue
        agg[k] = float(np.mean(vals))
    return agg


def cross_validate_model(
    model, X: pd.DataFrame, y: pd.Series, task: str, cv_folds: int = 5, splitter=None
) -> Tuple[Dict[str, float], float]:
    """`splitter` overrides the default KFold/StratifiedKFold — pass an
    ordered TimeSeriesSplit for temporal data so folds never see the future."""
    per_fold: List[Dict[str, float]] = []
    effective_task, _ = infer_effective_task(task, y)
    if effective_task == "classification":
        # Ensure labels are integer-coded for estimators that reject string classes (e.g., XGBoost).
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Series(
                pd.factorize(pd.Series(y).astype(str), sort=True)[0],
                name=getattr(y, "name", None),
                index=getattr(y, "index", None),
            )
        splitter = splitter or StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=42
        )
        for train_idx, val_idx in splitter.split(X, y):
            m = clone(model)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = m.predict(X.iloc[val_idx])
            y_proba = None
            if hasattr(m, "predict_proba"):
                y_proba = m.predict_proba(X.iloc[val_idx])
            elif hasattr(m, "decision_function"):
                y_proba = m.decision_function(X.iloc[val_idx])
            per_fold.append(classification_metrics(y.iloc[val_idx], y_pred, y_proba))
    else:
        splitter = splitter or KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in splitter.split(X, y):
            m = clone(model)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = m.predict(X.iloc[val_idx])
            per_fold.append(regression_metrics(y.iloc[val_idx], y_pred))

    agg = _aggregate_metrics(per_fold)
    # Primary metric for ranking
    if effective_task == "classification":
        primary = agg.get("roc_auc", agg.get("f1", agg.get("accuracy", 0.0)))
    else:
        primary = agg.get("rmse", float("inf"))
    return agg, float(primary)


def build_leaderboard(
    models: List[Tuple[str, object]],
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv_folds: int = 3,
    splitter=None,
) -> List[LeaderboardEntry]:
    entries: List[LeaderboardEntry] = []
    for name, est in models:
        try:
            scores, primary = cross_validate_model(
                est, X, y, task=task, cv_folds=cv_folds, splitter=splitter
            )
        except Exception:
            # A model incompatible with this data (e.g. KNN with fewer samples
            # than neighbors) must not kill the whole leaderboard — skip it.
            continue
        entries.append(LeaderboardEntry(model_name=name, scores=scores, fit_time=0.0))
    # Sort by primary metric (higher roc_auc/f1/accuracy or lower rmse)
    # Detect based on available scores to handle fallback to regression.
    if any(
        "roc_auc" in e.scores or "f1" in e.scores or "accuracy" in e.scores
        for e in entries
    ):
        entries = sorted(
            entries,
            key=lambda e: e.scores.get(
                "roc_auc", e.scores.get("f1", e.scores.get("accuracy", 0))
            ),
            reverse=True,
        )
    else:
        entries = sorted(entries, key=lambda e: e.scores.get("rmse", float("inf")))
    return entries
