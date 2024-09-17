"""Adapters tests."""

from __future__ import annotations

from simpml.tabular.adapters_pool import ManipulateAdapter


def test_manipulate_adapter() -> None:
    """Test the `ManipulateAdapter` class."""
    assert hasattr(ManipulateAdapter, "manipulate")
