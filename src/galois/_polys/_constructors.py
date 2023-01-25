"""
A module containing polynomial constructors that will be monkey-patched to Poly(), Poly.Int(), Poly.Degrees(),
and Poly.Random() in polys/__init__.py.

This is done to separate code into related modules without having circular imports with Poly().
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Type

import numpy as np
from typing_extensions import Literal

from .._domains import Array
from ..typing import ArrayLike

if TYPE_CHECKING:
    from ._poly import Poly


def POLY(
    coeffs: ArrayLike,
    field: Type[Array] | None = None,
    order: Literal["desc", "asc"] = "desc",
) -> Poly:
    raise NotImplementedError


def POLY_DEGREES(
    degrees: Sequence[int] | np.ndarray,
    coeffs: ArrayLike | None = None,
    field: Type[Array] | None = None,
) -> Poly:
    raise NotImplementedError


def POLY_INT(integer: int, field: Type[Array] | None = None) -> Poly:
    raise NotImplementedError


def POLY_RANDOM(
    degree: int,
    seed: int | np.integer | np.random.Generator | None = None,
    field: Type[Array] | None = None,
) -> Poly:
    raise NotImplementedError
