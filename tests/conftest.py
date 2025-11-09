"""Test configuration and fixtures."""

import pytest
from pathlib import Path

@pytest.fixture
def data_dir() -> Path:
    """Return the test data directory."""
    return Path(__file__).parent / "data"