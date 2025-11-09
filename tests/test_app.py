"""Test version and package structure."""

from app import __version__

def test_version():
    """Test version is a string."""
    assert isinstance(__version__, str)
    
def test_version_format():
    """Test version follows semantic versioning."""
    parts = __version__.split('.')
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)