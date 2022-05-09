"""
A module that defines a mixin classes for Galois field arithmetic.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .._domains import _calculate, _lookup
from .._domains._meta import DTYPES
from .._domains._ufunc import UFuncMixin


class UFuncMixin_2_m(UFuncMixin):
    """
    A mixin class that provides explicit calculation arithmetic for all GF(2^m) classes.
    """
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._add = _lookup.add_ufunc(cls, override=np.bitwise_xor)
        cls._negative = _lookup.negative_ufunc(cls, override=np.positive)
        cls._subtract = _lookup.subtract_ufunc(cls, override=np.bitwise_xor)
        cls._multiply = _calculate.multiply_binary(cls)
        cls._reciprocal = _calculate.reciprocal_fermat(cls)
        cls._divide = _calculate.divide(cls)
        cls._power = _calculate.field_power_square_and_multiply(cls)
        cls._log = _calculate.log_brute_force(cls)
        cls._sqrt = _calculate.sqrt_binary(cls)


class UFuncMixin_p_1(UFuncMixin):
    """
    A mixin class that provides explicit calculation arithmetic for all GF(p) classes.
    """
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._add = _calculate.add_modular(cls, always_calculate=True)
        cls._negative = _calculate.negative_modular(cls, always_calculate=True)
        cls._subtract = _calculate.subtract_modular(cls, always_calculate=True)
        cls._multiply = _calculate.multiply_modular(cls)
        cls._reciprocal = _calculate.reciprocal_modular_egcd(cls)
        cls._divide = _calculate.divide(cls)
        cls._power = _calculate.field_power_square_and_multiply(cls)
        cls._log = _calculate.log_brute_force(cls)
        cls._sqrt = _calculate.sqrt(cls)

    @classmethod
    def _determine_dtypes(cls) -> List[np.dtype]:
        """
        The only valid dtypes are ones that can hold x*x for x in [0, order).
        """
        max_dtype = DTYPES[-1]
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes


class UFuncMixin_p_m(UFuncMixin):
    """
    A mixin class that provides explicit calculation arithmetic for all GF(p^m) classes.
    """
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._add = _calculate.add_vector(cls)
        cls._negative = _calculate.negative_vector(cls)
        cls._subtract = _calculate.subtract_vector(cls)
        cls._multiply = _calculate.multiply_vector(cls)
        cls._reciprocal = _calculate.reciprocal_fermat(cls)
        cls._divide = _calculate.divide(cls)
        cls._power = _calculate.field_power_square_and_multiply(cls)
        cls._log = _calculate.log_brute_force(cls)
        cls._sqrt = _calculate.sqrt(cls)

    @classmethod
    def _determine_dtypes(cls) -> List[np.dtype]:
        """
        The only valid dtypes are ones that can hold x*x for x in [0, order).
        """
        # TODO: Is this correct?
        max_dtype = DTYPES[-1]
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes
