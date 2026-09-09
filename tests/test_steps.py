"""Steps tests."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.tabular.steps_pool import NanColumnDropper, SmartImpute

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")


def test_smart_impute() -> None:
    """Test the `SmartImpute` class."""
    assert hasattr(SmartImpute, "fit")
    assert hasattr(SmartImpute, "transform")


@pytest.fixture
def nan_df() -> pd.DataFrame:
    """A DataFrame with columns exhibiting different NaN fractions.

    Returns:
        A DataFrame with an all-NaN column, a half-NaN column, and a NaN-free column.
    """
    return pd.DataFrame(
        {
            "all_nan": [None, None, None, None],
            "half_nan": [1, None, 3, None],
            "no_nan": [1, 2, 3, 4],
        }
    )


def test_nan_column_dropper_default_threshold_drops_only_all_nan_columns(
    nan_df: pd.DataFrame,
) -> None:
    """With the default threshold (1.0), only fully-NaN columns should be dropped."""
    dropper = NanColumnDropper()
    transformed = dropper.fit(nan_df).transform(nan_df)

    assert list(dropper.columns_to_drop_) == ["all_nan"]
    assert list(transformed.columns) == ["half_nan", "no_nan"]


def test_nan_column_dropper_partial_threshold_drops_columns_at_or_above_it(
    nan_df: pd.DataFrame,
) -> None:
    """Columns whose NaN fraction meets or exceeds the threshold should be dropped."""
    dropper = NanColumnDropper(threshold=0.5)
    transformed = dropper.fit(nan_df).transform(nan_df)

    assert list(dropper.columns_to_drop_) == ["all_nan", "half_nan"]
    assert list(transformed.columns) == ["no_nan"]


def test_nan_column_dropper_zero_threshold_drops_every_column(
    nan_df: pd.DataFrame,
) -> None:
    """A threshold of 0.0 drops every column, since every NaN fraction is >= 0.0.

    Even a column with zero NaNs has a NaN fraction of 0.0, which meets the threshold.
    """
    dropper = NanColumnDropper(threshold=0.0)
    transformed = dropper.fit(nan_df).transform(nan_df)

    assert list(dropper.columns_to_drop_) == ["all_nan", "half_nan", "no_nan"]
    assert list(transformed.columns) == []


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_nan_column_dropper_rejects_out_of_range_threshold(threshold: float) -> None:
    """Thresholds outside of [0, 1] should raise a ValueError."""
    with pytest.raises(ValueError):
        NanColumnDropper(threshold=threshold)


def test_nan_column_dropper_transform_before_fit_raises() -> None:
    """Calling transform before fit should raise a RuntimeError."""
    dropper = NanColumnDropper()
    with pytest.raises(RuntimeError):
        dropper.transform(pd.DataFrame({"a": [1, 2]}))
