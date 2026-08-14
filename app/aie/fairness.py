"""
Fairness checks: group-wise performance for sensitive columns + disparity flags.
Used by the Train page's fairness section.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


def group_classification_metrics(
    y_true, y_pred, group: pd.Series
) -> Dict[str, Dict[str, float]]:
    results = {}
    n_classes = pd.Series(y_true).dropna().nunique()
    avg = "binary" if n_classes <= 2 else "weighted"
    for g in group.dropna().unique():
        mask = (group == g).to_numpy()
        if mask.sum() == 0:
            continue
        yt = np.asarray(y_true)[mask]
        yp = np.asarray(y_pred)[mask]
        results[g] = {
            "f1": f1_score(yt, yp, average=avg, zero_division=0),
            "accuracy": accuracy_score(yt, yp),
        }
    return results


def group_regression_metrics(
    y_true, y_pred, group: pd.Series
) -> Dict[str, Dict[str, float]]:
    results = {}
    for g in group.dropna().unique():
        mask = (group == g).to_numpy()
        if mask.sum() == 0:
            continue
        yt = np.asarray(y_true)[mask]
        yp = np.asarray(y_pred)[mask]
        results[g] = {"rmse": float(np.sqrt(mean_squared_error(yt, yp)))}
    return results


def disparity_alerts(
    group_metrics: Dict[str, Dict[str, float]], metric: str, max_gap: float
) -> List[str]:
    alerts = []
    values = [m[metric] for m in group_metrics.values() if metric in m]
    if not values:
        return alerts
    gap = max(values) - min(values)
    if gap > max_gap:
        alerts.append(f"Gap {gap:.3f} in {metric} exceeds {max_gap:.3f}")
    return alerts
