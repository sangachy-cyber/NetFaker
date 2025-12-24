import sys
from pathlib import Path

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the types from preprocessing module
from preprocessing import WindowMetaRaw, WindowMetaRenamed, WindowMetaNormed


def test_types_defined():
    """Test that our type definitions exist"""
    assert WindowMetaRaw is not None
    assert WindowMetaRenamed is not None
    assert WindowMetaNormed is not None


if __name__ == "__main__":
    pytest.main([__file__])
