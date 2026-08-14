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


def test_apply_filters_numeric_range():
    df = pl.DataFrame({"sales": [10.0, 50.0, 90.0], "id": [1, 2, 3]})

    filtered = apply_filters(df, {"sales": (20.0, 95.0)})

    assert filtered.get_column("id").to_list() == [2, 3]


def test_apply_filters_year_membership():
    df = pl.DataFrame({"year": [2019, 2020, 2021, 2022], "v": [1, 2, 3, 4]})

    filtered = apply_filters(df, {"year": [2020, 2022]})

    assert filtered.get_column("v").to_list() == [2, 4]


def test_apply_filters_date_range():
    from datetime import date

    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2021, 6, 1), date(2023, 1, 1)],
            "v": [1, 2, 3],
        }
    )

    filtered = apply_filters(df, {"date": (date(2021, 1, 1), date(2022, 1, 1))})

    assert filtered.get_column("v").to_list() == [2]
