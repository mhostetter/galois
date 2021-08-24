"""
A module that contains a metaclass mixin that provides GF(p) arithmetic using explicit calculation.
"""
import numba
import numpy as np

from ._class import FieldClass, DirMeta
from ._dtypes import DTYPES

RECIPROCAL = lambda a, *args: 1 / a


class GFpMeta(FieldClass, DirMeta):
    """
    An metaclass for all GF(p) classes.
    """
    # pylint: disable=no-value-for-parameter

    # Need to have a unique cache of "calculate" functions for GF(p)
    _FUNC_CACHE_CALCULATE = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._prime_subfield = cls

        cls.compile(kwargs["compile"])

    @property
    def dtypes(cls):
        """
        The only valid dtypes are ones that can hold x*x for x in [0, order).
        """
        max_dtype = DTYPES[-1]
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(d) == 0:
            d = [np.object_]
        return d

    def _ufunc(cls, name):
        # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835548405.
        if name not in cls._ufuncs and cls.ufunc_mode == "jit-lookup" and name in ["add", "negative", "subtract"]:
            cls._ufuncs[name] = cls._ufunc_calculate(name)
        return super()._ufunc(name)

    def _set_globals(cls, name):
        super()._set_globals(name)
        global RECIPROCAL

        if name in ["divide", "power"]:
            RECIPROCAL = cls._func_calculate("reciprocal", reset=False)

    def _reset_globals(cls):
        super()._reset_globals()
        global RECIPROCAL

        RECIPROCAL = cls._func_python("reciprocal")

    ###############################################################################
    # Arithmetic functions using explicit calculation
    ###############################################################################

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        c = a + b
        if c >= CHARACTERISTIC:
            c -= CHARACTERISTIC
        return c

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            return 0
        else:
            return CHARACTERISTIC - a

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a >= b:
            return a - b
        else:
            return CHARACTERISTIC + a - b

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        return (a * b) % CHARACTERISTIC

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        s*x + t*y = gcd(x, y) = 1
        x = p
        y = a in GF(p)
        t = a**-1 in GF(p)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        r2, r1 = CHARACTERISTIC, a
        t2, t1 = 0, 1

        while r1 != 0:
            q = r2 // r1
            r2, r1 = r1, r2 - q*r1
            t2, t1 = t1, t2 - q*t1

        if t2 < 0:
            t2 += CHARACTERISTIC

        return t2

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            b_inv = RECIPROCAL(b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
            return (a * b_inv) % CHARACTERISTIC

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Square and Multiply Algorithm

        a^13 = (1) * (a)^13
            = (a) * (a)^12
            = (a) * (a^2)^6
            = (a) * (a^4)^3
            = (a * a^4) * (a^4)^2
            = (a * a^4) * (a^8)
            = result_m * result_s
        """
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif b < 0:
            a = RECIPROCAL(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
            b = abs(b)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while b > 1:
            if b % 2 == 0:
                result_s = (result_s * result_s) % CHARACTERISTIC
                b //= 2
            else:
                result_m = (result_m * result_s) % CHARACTERISTIC
                b -= 1

        result = (result_m * result_s) % CHARACTERISTIC

        return result

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        TODO: Replace this with more efficient algorithm
        a = α^m
        b is a primitive element of the field

        c = log(a, b)
        a = b^c
        """
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        # Naive algorithm
        ORDER = CHARACTERISTIC**DEGREE
        result = 1
        for i in range(0, ORDER - 1):
            if result == a:
                break
            result = (result * b) % CHARACTERISTIC

        return i
