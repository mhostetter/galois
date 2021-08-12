"""
A module that contains a metaclass mixin that provides generic Galois field arithmetic using explicit calculation.
"""
import numba
from numba import int64
import numpy as np

from ._properties import PropertiesMeta


class CalculateMeta(PropertiesMeta):
    """
    A mixin metaclass that provides Galois field arithmetic using explicit calculation.
    """
    # pylint: disable=no-value-for-parameter

    # Function signatures for JIT-compiled arithmetic functions
    _UNARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64))
    _BINARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64, int64))

    # Lookup table of ufuncs to unary/binary type needed for LookupMeta, CalculateMeta, etc
    _UFUNC_TYPE = {
        "add": "binary",
        "negative": "unary",
        "subtract": "binary",
        "multiply": "binary",
        "reciprocal": "unary",
        "divide": "binary",
        "power": "binary",
        "log": "binary",
    }

    ###############################################################################
    # Individual JIT arithmetic functions, pre-compiled (cached)
    ###############################################################################

    def _calculate_jit(cls, name):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    def _python_func(cls, name):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    ###############################################################################
    # Individual ufuncs, compiled on-demand
    ###############################################################################

    def _calculate_ufunc(cls, name):
        raise NotImplementedError

    def _python_ufunc(cls, name):  # pylint: disable=no-self-use
        function = eval(f"cls._{name}_python")
        if cls._UFUNC_TYPE[name] == "unary":
            return np.frompyfunc(function, 1, 1)
        else:
            return np.frompyfunc(function, 2, 1)

    ###############################################################################
    # Pure-python arithmetic methods using explicit calculation
    ###############################################################################

    def _add_python(cls, a, b):
        raise NotImplementedError

    def _negative_python(cls, a):
        raise NotImplementedError

    def _subtract_python(cls, a, b):
        raise NotImplementedError

    def _multiply_python(cls, a, b):
        raise NotImplementedError

    def _reciprocal_python(cls, a):
        raise NotImplementedError

    def _divide_python(cls, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            b_inv = cls._reciprocal_python(b)
            return cls._multiply_python(a, b_inv)

    def _power_python(cls, a, b):
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif b < 0:
            a = cls._reciprocal_python(a)
            b = abs(b)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while b > 1:
            if b % 2 == 0:
                result_s = cls._multiply_python(result_s, result_s)
                b //= 2
            else:
                result_m = cls._multiply_python(result_m, result_s)
                b -= 1

        result = cls._multiply_python(result_m, result_s)

        return result

    def _log_python(cls, a, b):
        """
        TODO: Replace this with a more efficient algorithm
        """
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        # Naive algorithm
        result = 1
        for i in range(0, cls.order - 1):
            if result == a:
                break
            result = cls._multiply_python(result, b)

        return i
