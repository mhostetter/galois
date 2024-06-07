"""
A module for for creating Array subclasses. These functions are provided to prevent circular dependencies.
They will be monkey-patched in galois/__init__.py.
"""

from typing import Type

from ._array import Array


def FIELD_FACTORY(*args, **kwargs) -> Type[Array]:  # pragma: no cover
    """
    This will be monkey-patched to be `galois.GF()` in galois/__init__.py.
    """
    return Array


DEFAULT_ARRAY = Array
"""
This will be monkey-patched to be `galois.GF2` in galois/__init__.py.
"""
