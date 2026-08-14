"""
Layer 2 â€“ Smart Data Diagnosis.

Produces a data health report prior to cleaning:
- Missing value summary and heatmap data (sampled).
- Duplicate rows.
- Outlier suspicion via IQR.
- Correlation hotspots (numeric) and categorical association (Cramer's V).
- Potential feature redundancy via high correlation.

Plots are returned as Plotly figures to keep UI simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# NeuroViz merge: Plotly figures removed â€” the UI renders the raw
# `matrix` / missing-mask data with Altair in the app's identity theme.


@dataclass
class MissingReport:
    per_column: pd.DataFrame
    heatmap_fig: Optional[object]


@dataclass
class DuplicateReport:
    duplicate_count: int
    duplicate_ratio: float


@dataclass
class OutlierReport:
    per_column: Dict[str, Dict[str, float]]
    method: str


@dataclass
class CorrelationHotspots:
    pairs: List[Tuple[str, str, float]]
    threshold: float
    matrix_fig: Optional[object]
    matrix: Optional[pd.DataFrame] = None


@dataclass
class CategoricalCorrelationHotspots:
    pairs: List[Tuple[str, str, float]]
    threshold: float
    matrix_fig: Optional[object]
    matrix: Optional[pd.DataFrame] = None


@dataclass
class DiagnosisSummary:
    missing: MissingReport
    duplicates: DuplicateReport
    outliers: OutlierReport
    correlations: CorrelationHotspots
    cat_correlations: Optional[CategoricalCorrelationHotspots] = None


def missing_values_report(df: pd.DataFrame, sample_rows: int = 200) -> MissingReport:
    per_col = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean() * 100,
        }
    ).sort_values("missing_pct", ascending=False)

    return MissingReport(per_column=per_col, heatmap_fig=None)


def duplicate_report(df: pd.DataFrame) -> DuplicateReport:
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    dup_ratio = dup_count / len(df) if len(df) else 0.0
    return DuplicateReport(duplicate_count=dup_count, duplicate_ratio=dup_ratio)


def outlier_report(
    df: pd.DataFrame, numeric_cols: List[str], iqr_multiplier: float = 1.5
) -> OutlierReport:
    per_col = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outlier_mask = (series < lower) | (series > upper)
        per_col[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_pct": float(outlier_mask.mean() * 100),
        }
    return OutlierReport(per_column=per_col, method=f"IQR x {iqr_multiplier}")


def correlation_hotspots(
    df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.8
) -> CorrelationHotspots:
    pairs: List[Tuple[str, str, float]] = []
    corr_fig = None
    valid_numeric = [c for c in numeric_cols if c in df.columns]
    corr_matrix = None
    if valid_numeric:
        corr_matrix = df[valid_numeric].corr().abs()
        for i, col_i in enumerate(numeric_cols):
            if col_i not in corr_matrix.columns:
                continue
            for col_j in numeric_cols[i + 1 :]:
                if col_j not in corr_matrix.columns:
                    continue
                val = corr_matrix.loc[col_i, col_j]
                if pd.notna(val) and val >= threshold:
                    pairs.append((col_i, col_j, float(val)))
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    return CorrelationHotspots(
        pairs=pairs_sorted, threshold=threshold, matrix_fig=corr_fig, matrix=corr_matrix
    )


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    # Bias-corrected Cramer's V for nominal association between two categorical variables.
    table = pd.crosstab(x, y)
    if table.empty:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(rcorr - 1, kcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def categorical_correlation_hotspots(
    df: pd.DataFrame, categorical_cols: List[str], threshold: float = 0.3
) -> CategoricalCorrelationHotspots:
    pairs: List[Tuple[str, str, float]] = []
    cat_cols = [c for c in categorical_cols if c in df.columns]
    corr_matrix = None
    corr_fig = None
    if len(cat_cols) >= 2:
        corr_data = np.ones((len(cat_cols), len(cat_cols)))
        for i, col_i in enumerate(cat_cols):
            for j, col_j in enumerate(cat_cols):
                if i >= j:
                    continue
                v = _cramers_v(df[col_i], df[col_j])
                corr_data[i, j] = v
                corr_data[j, i] = v
                if v >= threshold:
                    pairs.append((col_i, col_j, float(v)))
        corr_matrix = pd.DataFrame(corr_data, index=cat_cols, columns=cat_cols)
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    return CategoricalCorrelationHotspots(
        pairs=pairs_sorted,
        threshold=threshold,
        matrix_fig=corr_fig,
        matrix=corr_matrix,
    )


def diagnose(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
    corr_threshold: float = 0.8,
    cat_corr_threshold: float = 0.3,
    sample_rows: int = 200,
) -> DiagnosisSummary:
    missing = missing_values_report(df, sample_rows=sample_rows)
    duplicates = duplicate_report(df)
    outliers = outlier_report(
        df, numeric_cols=numeric_cols, iqr_multiplier=iqr_multiplier
    )
    correlations = correlation_hotspots(
        df, numeric_cols=numeric_cols, threshold=corr_threshold
    )
    cat_cols = categorical_cols or []
    cat_correlations = (
        categorical_correlation_hotspots(
            df, categorical_cols=cat_cols, threshold=cat_corr_threshold
        )
        if cat_cols
        else None
    )
    return DiagnosisSummary(
        missing=missing,
        duplicates=duplicates,
        outliers=outliers,
        correlations=correlations,
        cat_correlations=cat_correlations,
    )
