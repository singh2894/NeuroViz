import polars as pl
import streamlit as st


def unique_filter_options(df: pl.DataFrame, column: str) -> list[str]:
    """Return sorted, non-null, string representations for a column."""
    series = df.get_column(column)
    return series.cast(pl.Utf8, strict=False).drop_nulls().unique().sort().to_list()


def build_filters(df: pl.DataFrame) -> dict[str, list[str]]:
    """
    Build multiselect filters for each text/categorical column.
    Returns a dict: {column_name: [selected_values]}.
    """
    filters: dict[str, list[str]] = {}

    for col, dtype in zip(df.columns, df.dtypes, strict=False):
        # Only filter string/categorical columns
        if dtype in (pl.Utf8, pl.Categorical):
            options = unique_filter_options(df, col)

            selected = st.sidebar.multiselect(
                f"Filter {col}",
                options=options,
                default=[],
                key=f"filter_{col}",  # <- important: stable key
            )

            if selected:
                filters[col] = selected

    return filters


def apply_filters(df: pl.DataFrame, filters: dict[str, list[str]]) -> pl.DataFrame:
    """
    Apply filters to the DataFrame.
    """
    for col, selected_values in filters.items():
        if not selected_values:
            continue

        dtype = df.schema.get(col)
        col_expr = pl.col(col)
        if dtype == pl.Categorical:
            # Cast to string so Streamlit selections (strings) match categorical values.
            col_expr = col_expr.cast(pl.Utf8, strict=False)

        df = df.filter(col_expr.is_in(selected_values))
    return df
