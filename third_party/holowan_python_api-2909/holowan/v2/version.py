"""
HoloWAN Network Emulator
Python API
"""

import warnings
from typing import Callable

_DEPRECATION_REASON: str = r""


def deprecated(func: Callable):
    def wrapper(*args, **kwargs):
        warnings.warn(f"The old version of holowan python API will be deprecated soon. Use the new version instead.",
                      category=DeprecationWarning
                      )
        return func(*args, **kwargs)

    return wrapper
