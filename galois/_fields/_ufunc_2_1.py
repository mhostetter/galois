"""
A module that defines a mixin class GF(2) arithmetic.
"""
from __future__ import annotations

import numba
import numpy as np

from .._domains._ufunc import FieldUfuncs

MULTIPLY = lambda a, b, *args: a * b
RECIPROCAL = lambda a, *args: 1 / a


class FieldUfuncs_2_1(FieldUfuncs):
    """
    A mixin class that provides explicit calculation arithmetic for GF(2).
    """
    # Need to have a unique cache of "calculate" functions for GF(2)
    _FUNC_CACHE_CALCULATE = {}

    @classmethod
    def _reset_ufuncs(cls):
        super()._reset_ufuncs()
        assert cls.ufunc_mode == "jit-calculate"

        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["negative"] = np.positive
        cls._ufuncs["subtract"] = np.bitwise_xor
        cls._ufuncs["multiply"] = np.bitwise_and
        cls._ufuncs["reciprocal"] = np.positive
        cls._ufuncs["divide"] = np.bitwise_and

    @classmethod
    def _set_globals(cls, name: str):
        return

    @classmethod
    def _reset_globals(cls):
        return

    ###############################################################################
    # Override ufunc routines to use native numpy bitwise ufuncs for GF(2)
    # arithmetic, which is faster than custom ufuncs
    ###############################################################################

    @classmethod
    def _ufunc_routine_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, b in GF(2)
        b = 1 / a, a = 1 is the only valid element with a multiplicative inverse, which is 1
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        if np.count_nonzero(inputs[0]) != inputs[0].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        output = getattr(cls._ufunc("reciprocal"), method)(*inputs, **kwargs)
        return output

    @classmethod
    def _ufunc_routine_divide(cls, ufunc, method, inputs, kwargs, meta):
        """
        Need to re-implement this to manually throw ZeroDivisionError if necessary
        """
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        if np.count_nonzero(inputs[meta["operands"][-1]]) != inputs[meta["operands"][-1]].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        output = getattr(cls._ufunc("divide"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, c in GF(2)
        c = a ** 2
          = a * a
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        return inputs[0]

    ###############################################################################
    # Arithmetic functions using explicit calculation
    ###############################################################################

    @staticmethod
    def _add_calculate(a: int, b: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    def _negative_calculate(a: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        """
        Not actually used. `np.positive()` is faster.
        """
        return a

    @staticmethod
    def _subtract_calculate(a: int, b: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    def _multiply_calculate(a: int, b: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        """
        Not actually used. `np.bitwise_and()` is faster.
        """
        return a & b

    @staticmethod
    def _reciprocal_calculate(a: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        return 1

    @staticmethod
    def _divide_calculate(a: int, b: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        return a & b

    @staticmethod
    @numba.extending.register_jitable
    def _power_calculate(a: int, b: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1

        return a

    @staticmethod
    @numba.extending.register_jitable
    def _log_calculate(a: int, b: int, CHARACTERISTIC: int, DEGREE: int, IRREDUCIBLE_POLY: int) -> int:
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        if b != 1:
            raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")

        return 0

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a: FieldUfuncs_2_1) -> FieldUfuncs_2_1:
        return a.copy()
