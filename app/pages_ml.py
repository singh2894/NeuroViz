# app/pages_ml.py — ML pipeline pages (vendored Automated Insight Engine)

import pickle

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from app.aie import cleaning, diagnostics, evaluation, models, runner, understanding

MODEL_NAMES = {
    "log_reg": "Logistic Regression",
    "rf": "Random Forest",
    "knn": "K-Nearest Neighbors",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "svm": "Support Vector Classifier",
    "nb": "Naive Bayes",
    "mlp": "Neural Network",
    "lin_reg": "Linear Regression",
    "rf_reg": "Random Forest Regressor",
    "knn_reg": "KNN Regressor",
    "xgb_reg": "XGBoost Regressor",
    "lgbm_reg": "LightGBM Regressor",
    "svr": "Support Vector Regressor",
    "mlp_reg": "Neural Network Regressor",
}
ERROR_METRICS = {"rmse", "mae", "mape"}


def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Bridge into the ML engine: ±inf becomes NaN so sklearn stages
    treat it as missing data instead of crashing."""
    return df.to_pandas().replace([np.inf, -np.inf], np.nan)


def _ver() -> int:
    """Data version — bumped on every upload/save, part of every cache key."""
    return st.session_state.get("data_version", 0)


# Leading-underscore params are excluded from the cache key: the key is
# (dataset name, data version, target_hint), never the frame's contents.
@st.cache_data(show_spinner=False)
def _summarize_cached(name: str, version: int, target_hint, _pdf: pd.DataFrame):
    return understanding.summarize(_pdf, target_hint=target_hint)


@st.cache_data(show_spinner=False)
def _diagnose_cached(
    name: str,
    version: int,
    _pdf: pd.DataFrame,
    _numeric_cols: list,
    _cat_cols: list,
):
    return diagnostics.diagnose(
        _pdf, numeric_cols=_numeric_cols, categorical_cols=_cat_cols
    )


def _status(text: str, color: str = "#4A7C63") -> None:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;'
        'font-size:12px;color:#55554E;">'
        f'<span style="width:8px;height:8px;background:{color};'
        'display:inline-block;"></span>'
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def guarded(label: str, render, *args) -> None:
    """Universal safety net: a section that can't handle this dataset says
    so in one line instead of crashing the whole page."""
    try:
        render(*args)
    except Exception as exc:
        _status(
            f"{label} unavailable for this dataset — " f"{type(exc).__name__}: {exc}",
            "#A8663F",
        )


def _heatmap(matrix: pd.DataFrame, value_title: str):
    long = matrix.reset_index(names="row").melt(
        id_vars="row", var_name="col", value_name="value"
    )
    return (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("col", title=None),
            y=alt.Y("row", title=None),
            color=alt.Color(
                "value",
                scale=alt.Scale(range=["#EAEAE5", "#1C1C1C"]),
                title=value_title,
            ),
            tooltip=["row", "col", alt.Tooltip("value", format=".2f")],
        )
        .properties(height=max(220, 24 * len(matrix)))
    )


def _bar(pdf: pd.DataFrame, x: str, y: str, ascending: bool = False):
    return (
        alt.Chart(pdf)
        .mark_bar()
        .encode(
            x=alt.X(x, sort="y" if ascending else "-y"),
            y=alt.Y(y, axis=alt.Axis(format="~s")),
            tooltip=[x, alt.Tooltip(y, format=",.4f")],
        )
        .properties(height=260)
    )


def _summary_for(name: str, pdf: pd.DataFrame, target_hint=None):
    """Understanding summary, cached per dataset — never shared across datasets."""
    return _summarize_cached(name, _ver(), target_hint, pdf)


def render_diagnostics_section(df: pl.DataFrame) -> None:
    guarded("Profile & diagnosis", _diagnostics_impl, df)


def _diagnostics_impl(df: pl.DataFrame) -> None:
    """Data page add-on: AIE understanding + diagnosis in identity styling."""
    name = st.session_state.get("dataset", "dataset")
    pdf = _to_pandas(df)
    target_hint = st.selectbox(
        "Target column (for task inference)",
        ["(auto-detect)"] + list(pdf.columns),
        key="ml_target_hint",
    )
    summary = _summary_for(
        name, pdf, None if target_hint == "(auto-detect)" else target_hint
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Schema")
        st.dataframe(
            pd.DataFrame(
                [
                    ("numeric", ", ".join(summary.schema.numeric) or "–"),
                    ("categorical", ", ".join(summary.schema.categorical) or "–"),
                    ("datetime", ", ".join(summary.schema.datetime) or "–"),
                    ("boolean", ", ".join(summary.schema.boolean) or "–"),
                ],
                columns=["kind", "columns"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.subheader("Target & task")
        _status(
            f"target={summary.target.target or 'none'} · "
            f"task={summary.task.task or 'unknown'}"
        )
        st.caption(f"Read as: {summary.target.reason} {summary.task.reason}")
        if summary.imbalance and summary.imbalance.is_imbalanced:
            _status(f"class imbalance · ratio={summary.imbalance.ratio:.2f}", "#8C8340")
        if summary.skew.skewed_features:
            _status("skewed: " + ", ".join(summary.skew.skewed_features[:6]), "#8C8340")
        if summary.leakage.suspicious_features:
            _status(
                "possible leakage: "
                + ", ".join(summary.leakage.suspicious_features[:6]),
                "#A8663F",
            )

    low_card_cats = [c for c in summary.schema.categorical if pdf[c].nunique() <= 100]
    diag = _diagnose_cached(name, _ver(), pdf, summary.schema.numeric, low_card_cats)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Missing values")
        missing = diag.missing.per_column
        st.dataframe(missing[missing["missing_count"] > 0], use_container_width=True)
        if pdf.isna().any().any():
            mask = pdf.head(200).isna().astype(int)
            mask.index.name = "row"
            st.altair_chart(
                _heatmap(mask.T, "missing").properties(height=260),
                use_container_width=True,
                theme=None,
            )
            st.caption("Missing-value pattern, first 200 rows")
        else:
            _status("no missing values")
    with c4:
        st.subheader("Duplicates & outliers")
        _status(
            f"{diag.duplicates.duplicate_count} duplicate rows "
            f"({diag.duplicates.duplicate_ratio:.1%})",
            "#4A7C63" if diag.duplicates.duplicate_count == 0 else "#8C8340",
        )
        if diag.outliers.per_column:
            st.dataframe(
                pd.DataFrame(diag.outliers.per_column).T,
                use_container_width=True,
            )
            st.caption(f"Outlier suspicion via {diag.outliers.method}")
        else:
            _status("no outliers flagged")

    if diag.correlations.matrix is not None and len(diag.correlations.matrix) >= 2:
        st.subheader("Correlation hotspots")
        st.altair_chart(
            _heatmap(diag.correlations.matrix, "|corr|"),
            use_container_width=True,
            theme=None,
        )
        if diag.correlations.pairs:
            st.caption(
                "Above threshold: "
                + " · ".join(
                    f"{a}×{b}={v:.2f}" for a, b, v in diag.correlations.pairs[:8]
                )
            )

    st.subheader("Categorical association (Cramér's V)")
    cat_corr = diag.cat_correlations
    if (
        cat_corr is not None
        and cat_corr.matrix is not None
        and len(cat_corr.matrix) >= 2
    ):
        st.altair_chart(
            _heatmap(cat_corr.matrix, "V"),
            use_container_width=True,
            theme=None,
        )
        if cat_corr.pairs:
            st.caption(
                "Strongly associated: "
                + " · ".join(f"{a}×{b}={v:.2f}" for a, b, v in cat_corr.pairs[:8])
            )
        else:
            st.caption(
                "0 = independent, 1 = perfectly associated — no pair exceeds "
                f"the {cat_corr.threshold} threshold"
            )
    else:
        # Never vanish silently — say why there's nothing to draw.
        n_cats = len(low_card_cats)
        _status(
            f"needs at least 2 categorical columns (≤100 unique values) — "
            f"this dataset has {n_cats}",
            "#8C8340",
        )


# -----------------------------------------------------
# Diagnose page
# -----------------------------------------------------
def diagnose_page():
    from app.main_app import select_dataset

    st.title("Diagnose")
    st.caption(
        "What the data contains and what's wrong with it — missing values, "
        "duplicates, outliers, correlated columns."
    )
    df = select_dataset()
    if df is None:
        return
    render_diagnostics_section(df)


# -----------------------------------------------------
# Clean page
# -----------------------------------------------------
def clean_page():
    from app.main_app import select_dataset

    st.title("Clean")
    st.caption("Fix what Diagnose found: fill gaps, drop junk columns, cap outliers.")
    df = select_dataset()
    if df is None:
        return
    guarded("Cleaning", _clean_impl, df)


def _clean_impl(df: pl.DataFrame) -> None:
    from app.main_app import get_datasets

    name = st.session_state.get("dataset", "dataset")
    pdf = _to_pandas(df)
    summary = _summary_for(name, pdf)

    sparse_threshold = st.slider("Drop columns missing more than", 0.0, 1.0, 0.9, 0.05)
    # Text columns that are really numbers ("1,234", "N/A", "?") get
    # coerced to numeric — vague values become NaN and are then imputed.
    convertible = [
        c
        for c in getattr(summary.schema, "convertible_text_numeric", [])
        if c in pdf.columns
    ]
    if convertible:
        _status(f"coercing text→numeric: {', '.join(convertible[:8])}")
    cleaned, info = cleaning.clean_dataset(
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
        iqr_multiplier=1.5,
        sparse_threshold=sparse_threshold,
    )
    if info.dropped_duplicates:
        _status(f"removed {info.dropped_duplicates} duplicate rows")
    if info.dropped_constant:
        _status(f"dropped constant: {', '.join(info.dropped_constant[:8])}", "#8C8340")
    if info.dropped_sparse:
        _status(f"dropped sparse: {', '.join(info.dropped_sparse[:8])}", "#8C8340")
    _status(
        f"imputation · numeric={info.imputation.get('numeric', '–')} · "
        f"categorical={info.imputation.get('categorical', '–')}"
    )
    if info.outlier_capped:
        _status(f"capped outliers in {len(info.outlier_capped)} columns")
    for note in info.notes:
        _status(note, "#8C8340")

    st.caption(f"{len(cleaned):,} rows × {cleaned.shape[1]} columns after cleaning")
    st.dataframe(cleaned.head(10), use_container_width=True)

    if st.button("Save as new dataset →", type="primary", key="save_clean"):
        get_datasets()[f"{name} (cleaned)"] = pl.from_pandas(cleaned)
        st.session_state["dataset"] = f"{name} (cleaned)"
        st.session_state["data_version"] = _ver() + 1
        st.success(f"Saved as {name} (cleaned) — available on every page")


# -----------------------------------------------------
# Engineer page (feature engineering)
# -----------------------------------------------------
# Each transform is a Polars expression producing one new, auto-named column.
NUM_TRANSFORMS = {
    "log (log1p)": lambda c: pl.col(c).log1p().alias(f"log_{c}"),
    "square root": lambda c: pl.col(c).sqrt().alias(f"sqrt_{c}"),
    "square": lambda c: (pl.col(c) ** 2).alias(f"{c}_sq"),
    "z-score standardize": lambda c: (
        (pl.col(c) - pl.col(c).mean()) / pl.col(c).std()
    ).alias(f"{c}_z"),
    "min-max 0–1": lambda c: (
        (pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())
    ).alias(f"{c}_01"),
}
PAIR_TRANSFORMS = {
    "ratio A ÷ B": lambda a, b: (pl.col(a) / pl.col(b)).alias(f"{a}_per_{b}"),
    "product A × B": lambda a, b: (pl.col(a) * pl.col(b)).alias(f"{a}_x_{b}"),
    "difference A − B": lambda a, b: (pl.col(a) - pl.col(b)).alias(f"{a}_minus_{b}"),
}


def engineer_page():
    from app.main_app import select_dataset

    st.title("Engineer")
    st.caption(
        "Create new columns from the ones you have — math transforms, "
        "ratios, bins, encodings, date parts. New columns appear on every "
        "page: filters, charts, and models."
    )
    df = select_dataset()
    if df is None:
        return
    guarded("Feature engineering", _engineer_impl, df)


def _engineer_impl(df: pl.DataFrame) -> None:
    from app.main_app import get_datasets

    name = st.session_state.get("dataset", "dataset")
    cols_types = list(zip(df.columns, df.dtypes, strict=False))
    numeric = [c for c, t in cols_types if t.is_numeric()]
    cats = [c for c, t in cols_types if t in (pl.Utf8, pl.Categorical)]
    dates = [c for c, t in cols_types if t in (pl.Date, pl.Datetime)] + [
        c for c, t in cols_types if t == pl.Utf8 and "date" in c.lower()
    ]

    options = (
        list(NUM_TRANSFORMS)
        + ["bins (quantile)"]
        + list(PAIR_TRANSFORMS)
        + ["one-hot encode", "category frequency", "date parts"]
    )
    choice = st.selectbox("Transform", options)

    result = None
    if choice in NUM_TRANSFORMS or choice == "bins (quantile)":
        if not numeric:
            st.info("This transform needs a numeric column.")
            return
        col = st.selectbox("Column", numeric)
        if choice == "bins (quantile)":
            n = st.slider("Number of bins", 2, 10, 4)
            result = df.with_columns(
                pl.col(col).qcut(n).cast(pl.Utf8).alias(f"{col}_bin")
            )
        else:
            result = df.with_columns(NUM_TRANSFORMS[choice](col))
    elif choice in PAIR_TRANSFORMS:
        if len(numeric) < 2:
            st.info("This transform needs two numeric columns.")
            return
        a = st.selectbox("Column A", numeric)
        b = st.selectbox("Column B", numeric, index=1)
        result = df.with_columns(PAIR_TRANSFORMS[choice](a, b))
    elif choice == "one-hot encode":
        if not cats:
            st.info("This transform needs a categorical column.")
            return
        col = st.selectbox("Column", cats)
        n_unique = df.get_column(col).n_unique()
        if n_unique > 20:
            st.warning(f"{col} has {n_unique} categories — too many to one-hot.")
            return
        result = df.hstack(df.select(col).to_dummies())
    elif choice == "category frequency":
        if not cats:
            st.info("This transform needs a categorical column.")
            return
        col = st.selectbox("Column", cats)
        result = df.with_columns(pl.len().over(col).alias(f"{col}_count"))
    elif choice == "date parts":
        if not dates:
            st.info("No date-like columns found in this dataset.")
            return
        col = st.selectbox("Column", dates)
        d = pl.col(col)
        if df.get_column(col).dtype == pl.Utf8:
            d = d.str.to_datetime(strict=False)
        result = df.with_columns(
            d.dt.year().alias(f"{col}_year"),
            d.dt.month().alias(f"{col}_month"),
            d.dt.weekday().alias(f"{col}_weekday"),
        )

    if result is None:
        return
    new_cols = [c for c in result.columns if c not in df.columns]
    if not new_cols:
        st.info("This transform is already applied — the columns exist.")
        return

    st.caption("Preview — new columns on the right")
    source_cols = [c for c in df.columns if any(c in n for n in new_cols)][:2]
    st.dataframe(
        result.select(source_cols + new_cols).head(10).to_pandas(),
        use_container_width=True,
    )

    if st.button(f"Add to {name} →", type="primary"):
        get_datasets()[name] = result
        st.session_state["data_version"] = _ver() + 1
        st.success(f"Added {', '.join(new_cols)} — available on every page")


# -----------------------------------------------------
# Features page
# -----------------------------------------------------
def features_page():
    from app.main_app import select_dataset

    st.title("Features")
    st.caption(
        "Which columns matter most for predicting the target — ranked across "
        "filter + embedded methods (variance, mutual information, model-based)."
    )
    df = select_dataset()
    if df is None:
        return
    guarded("Feature ranking", _features_impl, df)


def _features_impl(df: pl.DataFrame) -> None:
    name = st.session_state.get("dataset", "dataset")
    pdf = _to_pandas(df)
    summary = _summary_for(name, pdf)

    target = summary.target.target
    if not target or target not in pdf.columns:
        target = st.selectbox("Target column", pdf.columns, key="feat_target")
    task = summary.task.task or "regression"
    try:
        from app.aie import selection

        X, y = runner.prepare_features(pdf.dropna(subset=[target]), target)
        effective_task, _ = evaluation.infer_effective_task(task, y)
        scores = selection.filter_methods(X, y, effective_task)
        scores.update(selection.embedded_methods(X, y, effective_task))
        unified = selection.unify_importances(scores)
        imp = (
            pd.DataFrame(
                {"feature": list(unified), "importance": list(unified.values())}
            )
            .sort_values("importance", ascending=False)
            .head(20)
        )
        st.altair_chart(
            _bar(imp, "feature", "importance"),
            use_container_width=True,
            theme=None,
        )
        st.caption(f"Read as: unified feature importance for target={target}")
    except Exception as exc:
        st.info(f"Feature ranking unavailable for this data: {exc}")


# -----------------------------------------------------
# Model pages: Recommend / Train
# -----------------------------------------------------
def _model_setup(df: pl.DataFrame):
    """Shared prep for the model pages: target choice (remembered across
    pages), effective task, features, candidate models, and — when the data
    has a datetime column — an opt-in time ordering so validation never
    trains on the future."""
    name = st.session_state.get("dataset", "dataset")
    pdf = _to_pandas(df)
    summary = _summary_for(name, pdf)

    cols = list(pdf.columns)
    default_target = st.session_state.get("ml_target_choice") or summary.target.target
    target = st.selectbox(
        "Target column",
        cols,
        index=cols.index(default_target) if default_target in cols else 0,
    )
    st.session_state["ml_target_choice"] = target

    time_col = None
    date_cols = [c for c in summary.schema.datetime if c != target]
    if date_cols:
        time_col = st.selectbox(
            "Time-aware validation (train on the past, test on the future)",
            ["(off — random split)"] + date_cols,
            key="ml_time_col",
        )
        time_col = None if time_col == "(off — random split)" else time_col
    if time_col:
        pdf = pdf.sort_values(time_col)

    effective_task, _ = evaluation.infer_effective_task(
        summary.task.task or "classification", pdf[target]
    )
    if effective_task == "classification":
        n_unique = pdf[target].nunique()
        if n_unique > 50 and n_unique / max(len(pdf), 1) > 0.25:
            _status(
                f"target has {n_unique} unique values — treating as regression",
                "#8C8340",
            )
            effective_task = "regression"
    _status(
        f"task={effective_task} · target={target}"
        + (f" · ordered by {time_col}" if time_col else "")
    )

    X, y = runner.prepare_features(pdf.dropna(subset=[target]), target)
    label_classes: list[str] = []
    if effective_task == "classification":
        y, label_classes = runner.encode_classification_target(y)

    candidates = models.select_models(
        task=effective_task,
        n_rows=len(pdf),
        n_features=X.shape[1],
        n_categorical=len(summary.schema.categorical),
    )
    return name, pdf, target, effective_task, X, y, label_classes, candidates, time_col


def model_recommend_page():
    from app.main_app import select_dataset

    st.title("Recommend")
    st.caption(
        "Models suited to this data, then a cross-validated leaderboard to "
        "rank them."
    )
    df = select_dataset()
    if df is None:
        return
    guarded("Model recommendation", _recommend_impl, df)


def _recommend_impl(df: pl.DataFrame) -> None:
    name, pdf, target, effective_task, X, y, _, candidates, time_col = _model_setup(df)

    st.subheader("Candidates")
    st.caption(
        f"Candidates for {effective_task} on {len(pdf):,} rows × "
        f"{X.shape[1]} features (one-hot encoded + imputed inside each model "
        "pipeline, so validation folds never leak)"
    )
    for spec in candidates:
        with st.expander(MODEL_NAMES.get(spec.name, spec.name)):
            st.write(f"Estimator: `{type(spec.estimator).__name__}`")
            st.json(spec.estimator.get_params(), expanded=False)

    st.subheader("Leaderboard")
    cv_folds = st.slider("Cross-validation folds", 2, 5, 3)
    if st.button("Run leaderboard", type="primary"):
        splitter = TimeSeriesSplit(n_splits=cv_folds) if time_col else None
        with st.spinner(f"cross-validating {len(candidates)} models..."):
            lb = evaluation.build_leaderboard(
                [(s.name, runner.make_model(s.estimator, X)) for s in candidates],
                X,
                y,
                task=effective_task,
                cv_folds=cv_folds,
                splitter=splitter,
            )
        st.session_state["ml_leaderboard"] = {
            "dataset": name,
            "target": target,
            "task": effective_task,
            "lb": lb,
        }

    stored = st.session_state.get("ml_leaderboard")
    if stored and stored["dataset"] == name and stored["target"] == target:
        lb = stored["lb"]
        rows = [
            {"model": MODEL_NAMES.get(e.model_name, e.model_name)}
            | {k: round(v, 4) for k, v in e.scores.items()}
            for e in lb
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        best = MODEL_NAMES.get(lb[0].model_name, lb[0].model_name)
        _status(f"best model: {best} · {cv_folds}-fold cross-validation")

        metrics = (
            ["accuracy", "f1", "precision", "recall"]
            if stored["task"] == "classification"
            else ["r2", "rmse", "mae"]
        )
        metrics = [m for m in metrics if m in lb[0].scores]
        slots: list = []
        for i, metric in enumerate(metrics):
            if i % 2 == 0:
                slots = list(st.columns(2))
            plot = pd.DataFrame(
                {
                    "model": [MODEL_NAMES.get(e.model_name, e.model_name) for e in lb],
                    metric: [e.scores.get(metric) for e in lb],
                }
            ).dropna()
            with slots[i % 2]:
                st.altair_chart(
                    _bar(plot, "model", metric, ascending=metric in ERROR_METRICS),
                    use_container_width=True,
                    theme=None,
                )
                st.caption(f"Validation {metric} by model (CV mean)")


def model_train_page():
    from app.main_app import select_dataset

    st.title("Train")
    st.caption("Train the chosen model, then test it on rows it has never seen.")
    df = select_dataset()
    if df is None:
        return
    guarded("Training", _train_impl, df)


def _train_impl(df: pl.DataFrame) -> None:
    (
        name,
        _,
        target,
        effective_task,
        X,
        y,
        label_classes,
        candidates,
        time_col,
    ) = _model_setup(df)

    keys = [s.name for s in candidates]
    stored = st.session_state.get("ml_leaderboard")
    best_key = stored["lb"][0].model_name if stored else keys[0]
    selected = st.selectbox(
        "Model",
        keys,
        index=keys.index(best_key) if best_key in keys else 0,
        format_func=lambda k: MODEL_NAMES.get(k, k),
    )
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, 0.05)

    if st.button("Run train/test", type="primary"):
        if time_col:
            # Ordered split: the past trains, the future tests.
            cut = int(len(X) * train_frac)
            X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
            y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
        else:
            stratify = (
                y
                if effective_task == "classification" and y.value_counts().min() >= 2
                else None
            )
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, train_size=train_frac, random_state=42, stratify=stratify
            )
        spec = next(s for s in candidates if s.name == selected)
        model = runner.make_model(spec.estimator, X)
        with st.spinner("training..."):
            model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        if effective_task == "classification":
            proba = (
                model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
            )
            scores = evaluation.classification_metrics(y_te, y_pred, proba)
        else:
            scores = evaluation.regression_metrics(y_te, y_pred)

        try:
            with st.spinner("computing feature importance..."):
                perm = permutation_importance(
                    model, X_te, y_te, n_repeats=5, random_state=42
                )
            importance = (
                pd.DataFrame(
                    {"feature": X_te.columns, "importance": perm.importances_mean}
                )
                .sort_values("importance", ascending=False)
                .head(15)
            )
        except Exception:
            importance = None

        # Everything below renders from this stash, so uploading a file to
        # score (which reruns the script) keeps the trained model.
        st.session_state["ml_trained"] = {
            "dataset": name,
            "target": target,
            "task": effective_task,
            "model_key": selected,
            "model": model,
            "features": list(X.columns),
            "label_classes": label_classes,
            "scores": scores,
            "X_te": X_te,
            "y_te": y_te,
            "y_pred": pd.Series(y_pred, index=X_te.index),
            "importance": importance,
            "time_col": time_col,
            "n_train": len(X_tr),
        }

    run = st.session_state.get("ml_trained")
    if run and run["dataset"] == name and run["target"] == target:
        _render_train_results(run)


def _render_train_results(run: dict) -> None:
    scores, y_te, y_pred = run["scores"], run["y_te"], run["y_pred"]
    label_classes = run["label_classes"]
    split_note = (
        f"ordered by {run['time_col']} — tested on the most recent rows"
        if run["time_col"]
        else "random held-out split"
    )
    _status(
        f"{MODEL_NAMES.get(run['model_key'], run['model_key'])} · trained on "
        f"{run['n_train']:,} rows · tested on {len(y_te):,} rows · {split_note}"
    )
    mcols = st.columns(min(4, len(scores)))
    for slot, (k, v) in zip(mcols, list(scores.items())[:4], strict=False):
        slot.metric(k, f"{v:.4f}", border=True)

    c1, c2 = st.columns(2)
    with c1:
        if run["task"] == "classification":
            labels = list(range(len(label_classes))) or None
            cm = confusion_matrix(y_te, y_pred, labels=labels)
            names = [str(c) for c in label_classes] or [
                str(i) for i in range(cm.shape[0])
            ]
            cm_df = pd.DataFrame(cm, index=names, columns=names)
            st.altair_chart(
                _heatmap(cm_df, "count"), use_container_width=True, theme=None
            )
            st.caption("Confusion matrix — rows actual, columns predicted")
        else:
            resid = pd.DataFrame({"actual": list(y_te), "predicted": list(y_pred)})
            st.altair_chart(
                alt.Chart(resid)
                .mark_circle(size=50, opacity=0.5)
                .encode(
                    x=alt.X("actual", scale=alt.Scale(zero=False)),
                    y=alt.Y("predicted", scale=alt.Scale(zero=False)),
                    tooltip=["actual", "predicted"],
                )
                .properties(height=300),
                use_container_width=True,
                theme=None,
            )
            st.caption("Predicted vs actual on the test split")
    with c2:
        if run["importance"] is not None:
            st.altair_chart(
                _bar(run["importance"], "feature", "importance"),
                use_container_width=True,
                theme=None,
            )
            st.caption("Permutation importance on the test split (5 repeats)")
        else:
            st.info("Feature importance unavailable for this model.")

    preview = pd.DataFrame(
        {
            "actual": (
                [label_classes[int(v)] for v in y_te] if label_classes else list(y_te)
            ),
            "predicted": (
                [label_classes[int(v)] for v in y_pred]
                if label_classes
                else list(y_pred)
            ),
        }
    )
    with st.expander("Sample predictions (test split)"):
        st.dataframe(preview.head(50), use_container_width=True)

    with st.expander("Fairness check — does it perform equally across groups?"):
        guarded("Fairness", _fairness_section, run)

    st.subheader("Use the model")
    guarded("Scoring", _score_new_data_section, run)
    st.download_button(
        "Download model (.pkl)",
        data=pickle.dumps(run["model"]),
        file_name=f"neuroviz_{run['model_key']}.pkl",
        mime="application/octet-stream",
    )
    st.caption(
        "The file is the full pipeline (encoding + model). Load with "
        "`pickle.load(open('model.pkl','rb'))` and call "
        "`.predict(df[feature_columns])` on raw rows."
    )


def _fairness_section(run: dict) -> None:
    from app.aie import fairness, understanding

    X_te, y_te, y_pred = run["X_te"], run["y_te"], run["y_pred"]
    group_cols = [c for c in X_te.columns if not pd.api.types.is_numeric_dtype(X_te[c])]
    if not group_cols:
        _status("no categorical columns to group by", "#8C8340")
        return
    default = next(
        (c for c in group_cols if c.lower() in understanding.DEFAULT_SENSITIVE_COLS),
        group_cols[0],
    )
    col = st.selectbox(
        "Group by", group_cols, index=group_cols.index(default), key="fair_col"
    )
    group = X_te[col].astype(str)
    if run["task"] == "classification":
        gm = fairness.group_classification_metrics(y_te, y_pred, group)
        metric, max_gap = "f1", 0.1
    else:
        gm = fairness.group_regression_metrics(y_te, y_pred, group)
        metric, max_gap = "rmse", 0.2 * abs(run["scores"].get("rmse", 1.0))
    if not gm:
        _status("not enough data per group", "#8C8340")
        return
    table = pd.DataFrame(gm).T.rename_axis(col).reset_index()
    table["n"] = table[col].map(group.value_counts())
    st.dataframe(table, use_container_width=True, hide_index=True)
    alerts = fairness.disparity_alerts(gm, metric=metric, max_gap=max_gap)
    if alerts:
        for a in alerts:
            _status(f"disparity: {a} — treat per-group results with care", "#A8663F")
    else:
        _status(f"no {metric} disparity above {max_gap:.3f} across {col}")


def _score_new_data_section(run: dict) -> None:
    from app.data_io import read_tabular

    st.caption(
        "Upload rows with the same columns (target not needed) — get "
        "predictions back as a CSV."
    )
    uploaded = st.file_uploader(
        "Rows to score",
        type=["csv", "xlsx", "xls", "parquet"],
        key="score_upload",
        label_visibility="collapsed",
    )
    if uploaded is None:
        return
    new_df = read_tabular(uploaded)
    new_pdf = _to_pandas(new_df)
    missing = [c for c in run["features"] if c not in new_pdf.columns]
    if missing:
        st.error(f"These columns are missing from the upload: {', '.join(missing)}")
        return
    feats = runner.epochize_datetimes(new_pdf[run["features"]])
    try:
        preds = run["model"].predict(feats)
    except Exception as exc:
        st.error(
            f"Could not score this file ({type(exc).__name__}: {exc}). "
            "Check that column types match the training data."
        )
        return
    out = new_pdf.copy()
    label_classes = run["label_classes"]
    out[f"predicted_{run['target']}"] = (
        [label_classes[int(v)] for v in preds] if label_classes else preds
    )
    _status(f"scored {len(out):,} rows")
    st.dataframe(out.head(20), use_container_width=True)
    st.download_button(
        "Download predictions (CSV)",
        data=out.to_csv(index=False).encode(),
        file_name="predictions.csv",
        mime="text/csv",
    )
