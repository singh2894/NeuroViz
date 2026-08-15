# app/compilers/altair_compile.py

from __future__ import annotations

import re

import altair as alt
import numpy as np
import pandas as pd
import polars as pl

from app.parsers.synonyms import METRIC_SYNONYMS

# Allow big datasets when exporting JSON
alt.data_transformers.disable_max_rows()

# "NeuroViz Identity": monochrome shell, hue reserved for data.
_MONO_FONT = "'JetBrains Mono', ui-monospace, monospace"
DATA_CATEGORICAL = ["#3F5D8C", "#4A7C63", "#A8663F", "#7A5A8C", "#8C8340", "#3F7C8C"]
_GREY_RAMP = ["#EAEAE5", "#C6C6BF", "#9A9A93", "#6E6E67", "#454540", "#1C1C1C"]


@alt.theme.register("neuroviz", enable=True)
def _neuroviz_theme():
    return {
        "config": {
            "background": "transparent",
            "font": _MONO_FONT,
            "range": {
                "category": DATA_CATEGORICAL,
                "ramp": _GREY_RAMP,
                "heatmap": _GREY_RAMP,
            },
            "axis": {
                "labelColor": "#7A7A74",
                "titleColor": "#55554E",
                "gridColor": "#ECECE7",
                "domainColor": "#DDDDD7",
                "tickColor": "#DDDDD7",
            },
            "legend": {"labelColor": "#55554E", "titleColor": "#7A7A74"},
            "view": {"stroke": None},
            "bar": {"color": "#3F5D8C"},
            "line": {"color": "#3F5D8C"},
            "area": {"color": "#3F5D8C"},
            "point": {"color": "#3F5D8C"},
            "circle": {"color": "#3F5D8C"},
            "rect": {"color": "#3F5D8C"},
        }
    }


AGG_FUNCTIONS = {
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "mean": lambda col: pl.col(col).mean(),
    "sum": lambda col: pl.col(col).sum(),
}


def infer_schema(df: pl.DataFrame) -> dict:
    date_cols: list[str] = []
    numeric_cols: list[str] = []
    cat_cols: list[str] = []
    year_cols: list[str] = []

    for c, dt in zip(df.columns, df.dtypes, strict=False):
        name = str(c)
        lower_name = name.lower()
        if dt in (pl.Date, pl.Datetime) or "date" in lower_name or "time" in lower_name:
            date_cols.append(c)
        if dt.is_numeric():  # every int/uint/float width, not a hard-coded list
            numeric_cols.append(c)
        if dt in (pl.Utf8, pl.Categorical, pl.Boolean):
            cat_cols.append(c)

        # wide year columns like 1997, 1998 or F1997, F1998
        if re.fullmatch(r"\d{4}", name) or re.fullmatch(r"F\d{4}", name):
            year_cols.append(c)

    return {
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "year_cols": year_cols,
    }


def build_agg_expr(metric_col: str, agg: str | None, default: str | None = None):
    """
    Return (expression, agg_label). `default` is used when agg is None.
    """
    choice = agg if agg in AGG_FUNCTIONS else default
    if choice is None:
        return None, None
    expr = AGG_FUNCTIONS[choice](metric_col).alias(metric_col)
    return expr, choice


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def detect_column_mentions(intent_text: str, df: pl.DataFrame, schema: dict):
    """
    Return dict with numeric and categorical columns mentioned in the text.
    Matching is simple: column words must appear in the text.
    """
    normalized_text = normalize_text(intent_text)
    tokens = set(normalized_text.split())
    numeric_matches: list[str] = []
    categorical_matches: list[str] = []

    for col in df.columns:
        col_norm = normalize_text(str(col))
        if not col_norm:
            continue
        col_tokens = col_norm.split()

        match = False
        if col_norm in normalized_text:
            match = True
        elif col_tokens and all(tok in tokens for tok in col_tokens):
            match = True

        if match:
            if col in schema["numeric_cols"]:
                numeric_matches.append(col)
            elif col in schema["categorical_cols"]:
                categorical_matches.append(col)

    return {"numeric": numeric_matches, "categorical": categorical_matches}


def pick_metric_col(
    df: pl.DataFrame, intent_metric: str | None, preferred: list[str] | None = None
) -> str | None:
    cols = df.columns
    numeric_cols = [
        c for c, dt in zip(cols, df.dtypes, strict=False) if dt.is_numeric()
    ]
    if not numeric_cols:
        return None

    # Highest priority: explicitly referenced columns
    if preferred:
        for pref in preferred:
            if pref in numeric_cols:
                return pref

    # 1) try to map the requested metric to a real numeric column
    if intent_metric:
        target_words = [intent_metric]
        target_words += METRIC_SYNONYMS.get(intent_metric, [])

        for c in numeric_cols:
            name = c.lower()
            if any(w in name for w in target_words):
                return c

    # 2) fallback: prefer numeric columns with nice names
    prefer = [
        "sales",
        "revenue",
        "temperature",
        "value",
        "amount",
        "profit",
        "qty",
        "count",
        "y",
    ]
    for c in numeric_cols:
        if any(p in c.lower() for p in prefer):
            return c

    # 3) last resort: first numeric column
    return numeric_cols[0]


def pick_category_col(
    schema: dict, df: pl.DataFrame, preferred: list[str] | None = None
) -> str | None:
    if preferred:
        for c in preferred:
            if c in df.columns:
                return c

    for c in schema["categorical_cols"]:
        if c in df.columns:
            return c
    return None


def melt_years(
    df: pl.DataFrame, schema: dict, metric_name: str = "value"
) -> pl.DataFrame:
    """Convert wide year columns (1997, 1998, ...) into a long format Year/Value."""
    year_cols = schema["year_cols"]
    if not year_cols:
        return df

    id_cols = [c for c in df.columns if c not in year_cols]
    melted = df.unpivot(
        index=id_cols,
        on=year_cols,
        variable_name="Year",
        value_name=metric_name,
    )
    # Try to turn Year into integer
    return melted.with_columns(
        pl.col("Year").str.replace_all("F", "").cast(pl.Int64, strict=False)
    )


def apply_agg(df: pl.DataFrame, x_col: str, y_col: str, agg: str | None):
    expr, _ = build_agg_expr(y_col, agg)
    if expr is None:
        return df
    return df.group_by(x_col).agg(expr)


GRAIN_FREQ = {"1d": "D", "1w": "W", "1mo": "M", "1y": "Y"}
GRAIN_LABEL = {"1d": "day", "1w": "week", "1mo": "month", "1y": "year"}


def trend_line(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    parse_dates: bool,
    agg: str | None = None,
    grain: str | None = None,
):
    pdf = df.to_pandas()
    if parse_dates:
        converted = pd.to_datetime(
            pdf[x_col],
            errors="coerce",
            dayfirst=True,
        )
        if converted.notna().any():
            pdf[x_col] = converted
        pdf = pdf.dropna(subset=[x_col])

    # Bucket dates to the requested grain ("monthly", "yearly", ...)
    if grain and pd.api.types.is_datetime64_any_dtype(pdf[x_col]):
        pdf[x_col] = pdf[x_col].dt.to_period(GRAIN_FREQ[grain]).dt.start_time
    if agg or grain:
        pdf = pdf.groupby(x_col, as_index=False)[y_col].agg(agg or "sum")

    pdf = pdf.sort_values(x_col)
    pdf = pdf.dropna(subset=[y_col])
    return (
        alt.Chart(pdf)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=x_col,
            y=alt.Y(y_col, axis=alt.Axis(format="~s")),
            tooltip=[x_col, alt.Tooltip(y_col, format=",.2f")],
        )
        .properties(height=420)
        .interactive()
    )


def build_category_chart(df: pl.DataFrame, intent, schema: dict, hints: dict):
    cat_col = pick_category_col(schema, df, hints.get("categorical"))
    if not cat_col:
        return None, "I couldn't find a categorical column for a comparison chart."

    metric_col = pick_metric_col(df, intent.metric, hints.get("numeric"))
    if metric_col is None:
        return None, "I couldn't find a numeric column to compare."

    grouped_expr, used_agg = build_agg_expr(metric_col, intent.agg, default="mean")
    grouped = df.group_by(cat_col).agg(grouped_expr).drop_nulls([metric_col])
    if grouped.is_empty():
        return None, "The categorical chart had no data after aggregation."

    pdf = grouped.to_pandas().sort_values(metric_col, ascending=False)
    total_cats = len(pdf)
    if total_cats > 20:
        pdf = pdf.head(20)

    sel = alt.selection_point(name="cat_sel", fields=[cat_col])
    chart = (
        alt.Chart(pdf)
        .mark_bar()
        .encode(
            x=alt.X(cat_col, sort="-y"),
            y=alt.Y(metric_col, axis=alt.Axis(format="~s")),
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.35)),
            tooltip=[cat_col, alt.Tooltip(metric_col, format=",.2f")],
        )
        .add_params(sel)
        .properties(height=420)
    )
    agg_label = used_agg or "mean"
    caption = f"{agg_label.title()} of {metric_col} by {cat_col}"
    if total_cats > 20:
        caption += f" (top 20 of {total_cats})"
    return chart, caption


def build_scatter_chart(
    df: pl.DataFrame, schema: dict, hints: dict, require_hints: bool = False
):
    hinted = [c for c in hints.get("numeric", []) if c in df.columns]
    numeric_cols = (
        hinted
        if len(hinted) >= 2
        else [c for c in schema["numeric_cols"] if c in df.columns]
    )
    if require_hints and len(hinted) < 2:
        return None, "I couldn't find both requested numeric columns."

    if len(numeric_cols) < 2:
        return None, "I couldn't find two numeric columns for a scatter plot."

    x_col, y_col = numeric_cols[:2]
    work = df.select([x_col, y_col]).drop_nulls()
    if work.is_empty():
        return None, "After dropping nulls the scatter plot had no rows."
    # Points can't be aggregated meaningfully — cap what reaches the browser.
    sampled = work.height > 20_000
    if sampled:
        work = work.sample(20_000, seed=0)
    pdf = work.to_pandas()

    chart = (
        alt.Chart(pdf)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X(x_col, axis=alt.Axis(format="~s"), scale=alt.Scale(zero=False)),
            y=alt.Y(y_col, axis=alt.Axis(format="~s"), scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip(x_col, format=",.2f"),
                alt.Tooltip(y_col, format=",.2f"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    caption = f"Scatter of {y_col} vs {x_col}"
    if sampled:
        caption += " (20,000-point sample)"
    return chart, caption


def build_distribution_chart(df: pl.DataFrame, intent, schema: dict, hints: dict):
    metric_col = pick_metric_col(df, intent.metric, hints.get("numeric"))
    if metric_col is None:
        return None, "I couldn't find a numeric column for a distribution chart."

    # Pre-bin server-side: the browser gets 30 bars, never the raw values.
    vals = (
        df.get_column(metric_col)
        .drop_nulls()
        .cast(pl.Float64, strict=False)
        .drop_nulls()
        .to_numpy()
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, "No data available to build a distribution chart."

    counts, edges = np.histogram(vals, bins=30)
    pdf = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "Count": counts})
    chart = (
        alt.Chart(pdf)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "bin_start",
                bin="binned",
                title=metric_col,
                axis=alt.Axis(format="~s"),
            ),
            x2="bin_end",
            y=alt.Y("Count", title="Count"),
            tooltip=[
                alt.Tooltip("bin_start", format="~s", title="from"),
                alt.Tooltip("bin_end", format="~s", title="to"),
                "Count",
            ],
        )
        .properties(height=420)
    )
    return chart, f"Distribution of {metric_col}"


def build_trend_chart(df: pl.DataFrame, intent, schema: dict, hints: dict):
    if schema["date_cols"]:
        x_col = schema["date_cols"][0]
        metric_col = pick_metric_col(df, intent.metric, hints.get("numeric"))
        if metric_col is None:
            return None, "I couldn't find a numeric column to plot."

        agg, grain, auto_agg = intent.agg, intent.time_grain, False
        if df.schema.get(x_col) in (pl.Date, pl.Datetime):
            # Real temporal dtype: aggregate in polars — the browser only
            # ever receives the final line's points.
            work = df.select([x_col, metric_col]).drop_nulls()
            if grain:
                work = work.with_columns(pl.col(x_col).dt.truncate(grain))
            if not agg and not grain and work.height > 50_000:
                agg, auto_agg = "mean", True
            if agg or grain:
                # Aggregating on raw timestamps groups per-second — bucket
                # Datetime axes to days unless the user chose a grain.
                if not grain and df.schema.get(x_col) == pl.Datetime:
                    work = work.with_columns(pl.col(x_col).dt.truncate("1d"))
                expr, _ = build_agg_expr(metric_col, agg, default="sum")
                work = work.group_by(x_col).agg(expr)
            chart = trend_line(work.sort(x_col), x_col, metric_col, parse_dates=False)
        else:
            # String dates: pandas parsing path, but only the two needed columns.
            chart = trend_line(
                df.select([x_col, metric_col]),
                x_col,
                metric_col,
                parse_dates=True,
                agg=agg,
                grain=grain,
            )

        caption = f"Trend of {metric_col} over {x_col}"
        if grain:
            caption += f" ({agg or 'sum'} per {GRAIN_LABEL[grain]})"
        elif auto_agg:
            caption += " (auto-aggregated: mean per date)"
        elif agg:
            caption += f" ({agg} per {x_col})"
        return chart, caption

    if schema["year_cols"]:
        working_df = melt_years(df, schema)
        x_col = "Year"
        metric_col = pick_metric_col(working_df, intent.metric, hints.get("numeric"))
        if metric_col is None:
            return None, "I couldn't find a numeric column to plot."
        working_df = apply_agg(working_df, x_col, metric_col, intent.agg)
        chart = trend_line(working_df, x_col, metric_col, parse_dates=False)
        caption = f"Trend of {metric_col} over {x_col}"
        if intent.agg:
            caption += f" ({intent.agg} per {x_col})"
        return chart, caption

    return None, "I couldn't find a date or year column to make a trend chart."


def compile(intent, df: pl.DataFrame):
    """
    Main entry point: takes an Intent and a Polars DataFrame,
    returns (Altair chart or None, caption string).
    """
    schema = infer_schema(df)
    hints = detect_column_mentions(intent.text, df, schema)

    # Columns the LLM parser resolved take priority over text matching
    for col in reversed(getattr(intent, "columns", [])):
        if col in schema["numeric_cols"]:
            if col in hints["numeric"]:
                hints["numeric"].remove(col)
            hints["numeric"].insert(0, col)
        elif col in schema["categorical_cols"]:
            if col in hints["categorical"]:
                hints["categorical"].remove(col)
            hints["categorical"].insert(0, col)

    scatter_priority = len(hints.get("numeric", [])) >= 2

    def hinted_scatter(df_, intent_, schema_, hints_):
        return build_scatter_chart(df_, schema_, hints_, require_hints=True)

    def scatter(df_, intent_, schema_, hints_):
        return build_scatter_chart(df_, schema_, hints_)

    def conditional_scatter(df_, intent_, schema_, hints_):
        if not scatter_priority:
            return (
                None,
                "Mention two numeric columns (e.g., 'BMI vs age')"
                " to draw a scatter plot.",
            )
        return build_scatter_chart(df_, schema_, hints_, require_hints=True)

    strategy_lookup = {
        "trend": [
            lambda df, i, s, h: build_trend_chart(df, i, s, h),
            lambda df, i, s, h: build_category_chart(df, i, s, h),
            scatter,
            lambda df, i, s, h: build_distribution_chart(df, i, s, h),
        ],
        "compare": [
            conditional_scatter,
            lambda df, i, s, h: build_category_chart(df, i, s, h),
            lambda df, i, s, h: build_trend_chart(df, i, s, h),
            scatter,
            lambda df, i, s, h: build_distribution_chart(df, i, s, h),
        ],
        "rank": [
            lambda df, i, s, h: build_category_chart(df, i, s, h),
            lambda df, i, s, h: build_trend_chart(df, i, s, h),
            scatter,
            lambda df, i, s, h: build_distribution_chart(df, i, s, h),
        ],
        "distribution": [
            lambda df, i, s, h: build_distribution_chart(df, i, s, h),
            lambda df, i, s, h: build_trend_chart(df, i, s, h),
            lambda df, i, s, h: build_category_chart(df, i, s, h),
            scatter,
        ],
    }

    tried_reason = "I could not determine a chart for this dataset."
    builders = strategy_lookup.get(intent.kind, strategy_lookup["trend"])
    for builder in builders:
        if builder is None:
            continue
        chart, caption = builder(df, intent, schema, hints)
        if chart is not None:
            return chart, caption
        tried_reason = caption

    return None, tried_reason
