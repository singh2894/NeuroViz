"""
Layer 1 – Data Understanding Engine.

Responsibilities:
- Infer schema: numeric/categorical/datetime/boolean/convertible text.
- Infer target column and task type (classification vs regression).
- Detect class imbalance.
- Detect skewed numeric features.
- Flag potential leakage (features highly correlated with target).

Outputs are lightweight dicts so they can feed the UI and later layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Default sensitive columns for fairness checks can be configured elsewhere.
DEFAULT_SENSITIVE_COLS = [
    "gender",
    "sex",
    "age",
    "race",
    "ethnicity",
    "income",
    "region",
]


@dataclass
class SchemaInfo:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    boolean: List[str]
    convertible_text_numeric: List[str]


@dataclass
class TargetInference:
    target: Optional[str]
    reason: str


@dataclass
class TaskInference:
    task: Optional[str]  # "classification" | "regression"
    reason: str
    target_dtype: str
    target_cardinality: int


@dataclass
class ImbalanceReport:
    is_imbalanced: bool
    ratio: Optional[float]
    class_counts: Optional[Dict[str, int]]
    threshold: float


@dataclass
class SkewReport:
    skewed_features: Dict[str, float]
    threshold: float


@dataclass
class LeakageReport:
    suspicious_features: List[str]
    method: str
    detail: Dict[str, float]


@dataclass
class UnderstandingSummary:
    schema: SchemaInfo
    target: TargetInference
    task: TaskInference
    imbalance: Optional[ImbalanceReport]
    skew: SkewReport
    leakage: LeakageReport


def infer_schema(df: pd.DataFrame) -> SchemaInfo:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    # Remove boolean columns from numeric list to avoid double counting.
    numeric_cols = [c for c in numeric_cols if c not in bool_cols]
    categorical_cols = []
    for c in df.columns:
        if c in numeric_cols or c in datetime_cols or c in bool_cols:
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(
            df[c]
        ):
            categorical_cols.append(c)

    convertible_text_numeric = []
    for c in categorical_cols:
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        ratio_numeric = coerced.notna().mean() if len(s) else 0
        # Slightly permissive threshold to catch mostly-numeric text columns.
        if ratio_numeric >= 0.6:
            convertible_text_numeric.append(c)

    return SchemaInfo(
        numeric=numeric_cols,
        categorical=categorical_cols,
        datetime=datetime_cols,
        boolean=bool_cols,
        convertible_text_numeric=convertible_text_numeric,
    )


def infer_target(
    df: pd.DataFrame, target_hint: Optional[str] = None
) -> TargetInference:
    if target_hint and target_hint in df.columns:
        return TargetInference(target=target_hint, reason="Provided by user.")

    # Heuristic: choose last column that is not an obvious ID and has >1 unique.
    candidates = []
    for col in df.columns[::-1]:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            continue
        if "id" in col.lower():
            continue
        candidates.append((col, nunique))

    # If there is only one non-ID candidate, consider it ambiguous and defer to user.
    if len(candidates) <= 1:
        return TargetInference(
            target=None,
            reason="Ambiguous; only one non-ID column with >1 unique value.",
        )

    chosen = candidates[0][0]
    return TargetInference(
        target=chosen, reason="Heuristic: last non-ID column with >1 unique value."
    )


def infer_task(
    df: pd.DataFrame, target_col: str, classification_cardinality_cutoff: int = 20
) -> TaskInference:
    series = df[target_col]
    dtype = str(series.dtype)
    cardinality = series.nunique(dropna=True)

    if pd.api.types.is_bool_dtype(series):
        return TaskInference(
            task="classification",
            reason="Boolean target",
            target_dtype=dtype,
            target_cardinality=cardinality,
        )

    if pd.api.types.is_numeric_dtype(series):
        if cardinality <= classification_cardinality_cutoff:
            return TaskInference(
                task="classification",
                reason=f"Numeric target with low cardinality ({cardinality} <= {classification_cardinality_cutoff}).",
                target_dtype=dtype,
                target_cardinality=cardinality,
            )
        return TaskInference(
            task="regression",
            reason=f"Numeric target with cardinality {cardinality} > {classification_cardinality_cutoff}.",
            target_dtype=dtype,
            target_cardinality=cardinality,
        )

    return TaskInference(
        task="classification",
        reason="Non-numeric target.",
        target_dtype=dtype,
        target_cardinality=cardinality,
    )


def detect_imbalance(y: pd.Series, threshold: float = 0.2) -> ImbalanceReport:
    counts = y.value_counts(dropna=True)
    if counts.empty:
        return ImbalanceReport(
            is_imbalanced=False, ratio=None, class_counts=None, threshold=threshold
        )

    min_class = counts.min()
    total = counts.sum()
    ratio = min_class / total if total > 0 else None
    is_imbalanced = bool(ratio is not None and ratio < threshold)
    return ImbalanceReport(
        is_imbalanced=is_imbalanced,
        ratio=ratio,
        class_counts=counts.to_dict(),
        threshold=threshold,
    )


def detect_skew(
    df: pd.DataFrame, numeric_cols: List[str], threshold: float = 1.0
) -> SkewReport:
    skewed = {}
    for col in numeric_cols:
        skew_val = df[col].dropna().skew()
        if pd.notna(skew_val) and abs(skew_val) >= threshold:
            skewed[col] = float(skew_val)
    return SkewReport(skewed_features=skewed, threshold=threshold)


def _factorize_series(s: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(s.fillna("__MISSING__"))
    return codes.astype(float)


def detect_leakage(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    corr_threshold: float = 0.9,
    mi_top_k: int = 5,
) -> LeakageReport:
    y = df[target_col]
    suspicious = {}

    if pd.api.types.is_numeric_dtype(y):
        target_numeric = y
        # Numeric feature correlations
        for col in numeric_cols:
            if col == target_col:
                continue
            corr = abs(df[col].corr(target_numeric))
            if pd.notna(corr) and corr >= corr_threshold:
                suspicious[col] = float(corr)
        method = f"abs(corr) >= {corr_threshold}"
    else:
        # Mutual information for categorical targets
        y_encoded = _factorize_series(y)
        mi_scores = {}
        for col in numeric_cols + categorical_cols:
            if col == target_col:
                continue
            x = df[col]
            # Leakage detection is advisory: any column MI can't handle
            # (tiny samples, degenerate distributions, ±inf) is skipped.
            try:
                if pd.api.types.is_numeric_dtype(x):
                    x = x.replace([np.inf, -np.inf], np.nan)
                    if x.notna().sum() == 0:
                        continue
                    mi = mutual_info_classif(
                        x.fillna(x.median()).to_frame(),
                        y_encoded,
                        discrete_features=False,
                    )[0]
                else:
                    x_enc = _factorize_series(x)
                    mi = mutual_info_classif(
                        x_enc.reshape(-1, 1), y_encoded, discrete_features=True
                    )[0]
            except Exception:
                continue
            mi_scores[col] = float(mi)
        # Keep top K
        suspicious = dict(
            sorted(mi_scores.items(), key=lambda kv: kv[1], reverse=True)[:mi_top_k]
        )
        method = f"mutual_info_classif top {mi_top_k}"

    return LeakageReport(
        suspicious_features=list(suspicious.keys()), method=method, detail=suspicious
    )


def summarize(
    df: pd.DataFrame,
    target_hint: Optional[str] = None,
    classification_cardinality_cutoff: int = 20,
    imbalance_threshold: float = 0.2,
    skew_threshold: float = 1.0,
    corr_threshold: float = 0.9,
    mi_top_k: int = 5,
) -> UnderstandingSummary:
    schema = infer_schema(df)
    target_info = infer_target(df, target_hint=target_hint)

    if not target_info.target:
        # No target inferred; return minimal summary.
        empty_task = TaskInference(
            task=None,
            reason="No target inferred.",
            target_dtype="unknown",
            target_cardinality=0,
        )
        empty_imbalance = None
        skew_report = SkewReport(skewed_features={}, threshold=skew_threshold)
        leakage_report = LeakageReport(suspicious_features=[], method="n/a", detail={})
        return UnderstandingSummary(
            schema,
            target_info,
            empty_task,
            empty_imbalance,
            skew_report,
            leakage_report,
        )

    target_col = target_info.target
    task_info = infer_task(df, target_col, classification_cardinality_cutoff)

    imbalance = None
    if task_info.task == "classification":
        imbalance = detect_imbalance(df[target_col], threshold=imbalance_threshold)

    skew_report = detect_skew(df, schema.numeric, threshold=skew_threshold)
    leakage_report = detect_leakage(
        df,
        target_col=target_col,
        numeric_cols=schema.numeric,
        categorical_cols=schema.categorical,
        corr_threshold=corr_threshold,
        mi_top_k=mi_top_k,
    )

    return UnderstandingSummary(
        schema, target_info, task_info, imbalance, skew_report, leakage_report
    )
