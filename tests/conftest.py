"""Test configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Return the test data directory."""
    return Path(__file__).parent / "data"
