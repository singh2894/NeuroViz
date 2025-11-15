import polars as pl


def infer_schema(df: pl.DataFrame):
    date_cols = [
        c
        for c, t in zip(df.columns, df.dtypes, strict=False)
        if t in (pl.Date, pl.Datetime) or "date" in c.lower()
    ]
    num_cols = [
        c
        for c, t in zip(df.columns, df.dtypes, strict=False)
        if t in (pl.Float64, pl.Int64)
    ]
    cat_cols = [
        c
        for c, t in zip(df.columns, df.dtypes, strict=False)
        if t in (pl.Utf8, pl.Categorical)
    ]
    return {
        "date_cols": date_cols,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    }
