"""
A module that contains a metaclass mixin that provides GF(2^m) arithmetic using explicit calculation.
"""
import numba
import numpy as np

from ._class import FieldClass, DirMeta

MULTIPLY = lambda a, b, *args: a * b
RECIPROCAL = lambda a, *args: 1 / a


class GF2mMeta(FieldClass, DirMeta):
    """
    A metaclass for all GF(2^m) classes.
    """
    # pylint: disable=no-value-for-parameter

    # Need to have a unique cache of "calculate" functions for GF(2^m)
    _FUNC_CACHE_CALCULATE = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._prime_subfield = kwargs["prime_subfield"]

        cls.compile(kwargs["compile"])

        # Determine if the irreducible polynomial is primitive
        if cls._is_primitive_poly is None:
            # TODO: Clean this up
            coeffs = cls.irreducible_poly.coeffs.view(np.ndarray)
            x = np.array(cls.primitive_element, dtype=cls.dtypes[-1], ndmin=1)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            cls._is_primitive_poly = cls._function_python("poly_evaluate")(coeffs, x, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)[0] == 0

    def _compile_ufuncs(cls):
        super()._compile_ufuncs()

        # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835552639.
        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["negative"] = np.positive
        cls._ufuncs["subtract"] = np.bitwise_xor

    def _set_globals(cls, name):
        global MULTIPLY, RECIPROCAL

        if name in ["reciprocal", "divide", "power", "log"]:
            MULTIPLY = cls._func_calculate("multiply")
        if name in ["divide", "power"]:
            RECIPROCAL = cls._func_calculate("reciprocal")

    def _reset_globals(cls):
        global MULTIPLY, RECIPROCAL

        MULTIPLY = cls._func_python("multiply")
        RECIPROCAL = cls._func_python("reciprocal")

    ###############################################################################
    # Arithmetic functions using explicit calculation
    ###############################################################################

    @staticmethod
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.positive()` is faster.
        """
        return a

    @staticmethod
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        a in GF(2^m), can be represented as a degree m-1 polynomial a(x) in GF(2)[x]
        b in GF(2^m), can be represented as a degree m-1 polynomial b(x) in GF(2)[x]
        p(x) in GF(2)[x] with degree m is the irreducible polynomial of GF(2^m)

        a * b = c
            = (a(x) * b(x)) % p(x) in GF(2)
            = c(x)
            = c
        """
        ORDER = CHARACTERISTIC**DEGREE

        # Re-order operands such that a > b so the while loop has less loops
        if b > a:
            a, b = b, a

        c = 0
        while b > 0:
            if b & 0b1:
                c ^= a  # Add a(x) to c(x)

            b >>= 1  # Divide b(x) by x
            a <<= 1  # Multiply a(x) by x
            if a >= ORDER:
                a ^= IRREDUCIBLE_POLY  # Compute a(x) % p(x)

        return c

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        From Fermat's Little Theorem:
        a in GF(p^m)
        a^(p^m - 1) = 1

        a * a^-1 = 1
        a * a^-1 = a^(p^m - 1)
            a^-1 = a^(p^m - 2)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        ORDER = CHARACTERISTIC**DEGREE
        exponent = ORDER - 2
        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while exponent > 1:
            if exponent % 2 == 0:
                result_s = MULTIPLY(result_s, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                exponent //= 2
            else:
                result_m = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                exponent -= 1

        result = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

        return result

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            b_inv = RECIPROCAL(b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
            return MULTIPLY(a, b_inv, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

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
                result_s = MULTIPLY(result_s, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                b //= 2
            else:
                result_m = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                b -= 1

        result = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

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
            result = MULTIPLY(result, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

        return i
