"""Feature-engineering transforms: each produces the expected new column."""

import polars as pl

from app.pages_ml import NUM_TRANSFORMS, PAIR_TRANSFORMS

DF = pl.DataFrame({"a": [1.0, 4.0, 9.0], "b": [2.0, 2.0, 2.0], "cat": ["x", "x", "y"]})


def test_numeric_transforms_add_one_column_each():
    for label, make in NUM_TRANSFORMS.items():
        out = DF.with_columns(make("a"))
        assert out.width == DF.width + 1, label
        assert out.height == DF.height, label


def test_numeric_transform_values():
    out = DF.with_columns(NUM_TRANSFORMS["square"]("a"))
    assert out.get_column("a_sq").to_list() == [1.0, 16.0, 81.0]
    out = DF.with_columns(NUM_TRANSFORMS["min-max 0–1"]("a"))
    assert out.get_column("a_01").to_list()[0] == 0.0
    assert out.get_column("a_01").to_list()[-1] == 1.0


def test_pair_transforms():
    out = DF.with_columns(PAIR_TRANSFORMS["ratio A ÷ B"]("a", "b"))
    assert out.get_column("a_per_b").to_list() == [0.5, 2.0, 4.5]
    out = DF.with_columns(PAIR_TRANSFORMS["difference A − B"]("a", "b"))
    assert out.get_column("a_minus_b").to_list() == [-1.0, 2.0, 7.0]


def test_bins_frequency_and_one_hot():
    out = DF.with_columns(pl.col("a").qcut(2).cast(pl.Utf8).alias("a_bin"))
    assert out.get_column("a_bin").n_unique() == 2
    out = DF.with_columns(pl.len().over("cat").alias("cat_count"))
    assert out.get_column("cat_count").to_list() == [2, 2, 1]
    out = DF.hstack(DF.select("cat").to_dummies())
    assert {"cat_x", "cat_y"} <= set(out.columns)
