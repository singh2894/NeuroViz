import polars as pl

from app.components.filters import apply_filters, unique_filter_options


def test_unique_filter_options_casts_categorical_to_plain_strings():
    df = pl.DataFrame(
        {
            "model": pl.Series(["Model X", "Model Y", "Model X"], dtype=pl.Categorical),
        }
    )

    options = unique_filter_options(df, "model")

    assert options == ["Model X", "Model Y"]
    assert all(isinstance(opt, str) for opt in options)


def test_apply_filters_handles_categorical_columns():
    df = pl.DataFrame(
        {
            "fuel": pl.Series(["Electric", "Petrol", "Electric"], dtype=pl.Categorical),
            "value": [1, 2, 3],
        }
    )

    filtered = apply_filters(df, {"fuel": ["Electric"]})

    assert filtered.shape == (2, 2)
    assert filtered.get_column("value").to_list() == [1, 3]
