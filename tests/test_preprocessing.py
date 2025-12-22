import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    WindowMetaRaw, 
    WindowMetaRenamed, 
    WindowMetaNormed
)


def test_types_defined():
    """Test that our type definitions exist"""
    assert hasattr(sys.modules[__name__], 'WindowMetaRaw')
    assert hasattr(sys.modules[__name__], 'WindowMetaRenamed')
    assert hasattr(sys.modules[__name__], 'WindowMetaNormed')


if __name__ == "__main__":
    pytest.main([__file__])