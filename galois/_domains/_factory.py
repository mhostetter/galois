"""
A module for for creating Array subclasses. These functions are provided to prevent circular dependencies.
They will be monkey-patched into the _polys module.
"""
from typing import Type

from ._array import Array


def FIELD_FACTORY(*args, **kwargs) -> Type[Array]:  # pylint: disable=unused-argument
    """
    This will be monkey-patched to be `galois.GF()` in __init__.py.
    """
    return Array


DEFAULT_FIELD_ARRAY = Array
"""
This will be monkey-patched to be `galois.GF2` in __init__.py.
"""
