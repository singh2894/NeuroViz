import polars as pl
import streamlit as st

# A filter value is either a list (membership: categories, years) or a
# (lo, hi) tuple (range: numeric and date columns).
FilterValue = list | tuple


def unique_filter_options(df: pl.DataFrame, column: str) -> list[str]:
    """Return sorted, non-null, string representations for a column."""
    series = df.get_column(column)
    return series.cast(pl.Utf8, strict=False).drop_nulls().unique().sort().to_list()


def _is_year_col(df: pl.DataFrame, col: str, dtype: pl.DataType) -> bool:
    if not dtype.is_integer():
        return False
    if "year" in col.lower():
        return True
    s = df.get_column(col).drop_nulls()
    if s.is_empty():
        return False
    return 1800 <= s.min() and s.max() <= 2100


def _clear_filters(keys: list[str]) -> None:
    for key in keys:
        st.session_state.pop(key, None)  # widgets fall back to their defaults


def build_filters(df: pl.DataFrame) -> dict[str, FilterValue]:
    """
    Build sidebar filters: multiselects for text columns, year pickers for
    year-like columns, range sliders for dates and numbers.
    Returns {column_name: selected} containing only active filters.
    """
    filters: dict[str, FilterValue] = {}
    keys: list[str] = []
    numeric_cols: list[tuple[str, pl.DataType]] = []

    for col, dtype in zip(df.columns, df.dtypes, strict=False):
        key = f"filter_{col}"

        if dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
            keys.append(key)
            selected = st.sidebar.multiselect(
                f"Filter {col}",
                options=unique_filter_options(df, col),
                default=[],
                key=key,
            )
            if selected:
                filters[col] = selected

        elif _is_year_col(df, col, dtype):
            keys.append(key)
            years = df.get_column(col).drop_nulls().unique().sort().to_list()
            selected = st.sidebar.multiselect(
                f"Filter {col}", options=years, default=[], key=key
            )
            if selected:
                filters[col] = selected

        elif dtype in (pl.Date, pl.Datetime):
            s = df.get_column(col).drop_nulls()
            if s.is_empty() or s.min() == s.max():
                continue
            keys.append(key)
            lo, hi = st.sidebar.slider(
                f"Filter {col}",
                min_value=s.min(),
                max_value=s.max(),
                value=(s.min(), s.max()),
                key=key,
            )
            if (lo, hi) != (s.min(), s.max()):
                filters[col] = (lo, hi)

        elif dtype.is_numeric():
            numeric_cols.append((col, dtype))

    # Numeric ranges live in an expander so wide tables stay tidy
    if numeric_cols:
        with st.sidebar.expander("Numeric ranges"):
            for col, _ in numeric_cols[:15]:
                s = df.get_column(col).drop_nulls()
                if s.is_empty() or s.min() == s.max():
                    continue
                key = f"filter_{col}"
                keys.append(key)
                lo, hi = st.slider(
                    col,
                    min_value=float(s.min()),
                    max_value=float(s.max()),
                    value=(float(s.min()), float(s.max())),
                    key=key,
                )
                if (lo, hi) != (float(s.min()), float(s.max())):
                    filters[col] = (lo, hi)
            if len(numeric_cols) > 15:
                st.caption(f"Showing first 15 of {len(numeric_cols)} numeric columns")

    if filters:
        # on_click runs before widgets render, so clearing state here is safe
        st.sidebar.button("Reset filters", on_click=_clear_filters, args=(keys,))

    return filters


def apply_filters(df: pl.DataFrame, filters: dict[str, FilterValue]) -> pl.DataFrame:
    """
    Apply filters to the DataFrame. Lists filter by membership,
    (lo, hi) tuples filter by range.
    """
    for col, selected in filters.items():
        if isinstance(selected, tuple):
            lo, hi = selected
            df = df.filter(pl.col(col).is_between(lo, hi))
            continue
        if not selected:
            continue

        col_expr = pl.col(col)
        if all(isinstance(v, str) for v in selected):
            # Streamlit selections are strings; cast so categoricals match
            col_expr = col_expr.cast(pl.Utf8, strict=False)
        df = df.filter(col_expr.is_in(selected))
    return df
