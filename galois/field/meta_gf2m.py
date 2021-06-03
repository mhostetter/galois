import numba
import numpy as np

from ..dtypes import DTYPES
from .meta import FieldMeta

CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
ORDER = None  # The field's order `p^m`
IRREDUCIBLE_POLY = None  # The field's primitive polynomial in integer form
PRIMITIVE_ELEMENT = None  # The field's primitive element in integer form

MULTIPLY_UFUNC = lambda x, y: x * y
RECIPROCAL_UFUNC = lambda x: 1 / x


class GF2mMeta(FieldMeta):
    """
    An abstract base class for all :math:`\\mathrm{GF}(2^m)` field array classes.
    """
    # pylint: disable=abstract-method,no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly_int = cls.irreducible_poly.integer  # pylint: disable=no-member
        cls._prime_subfield = kwargs["prime_subfield"]

        cls.compile(kwargs["mode"], kwargs["target"])

        # Determine if the irreducible polynomial is primitive
        cls._is_primitive_poly = cls._irreducible_poly(cls.primitive_element, field=cls) == 0

    @property
    def dtypes(cls):
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1]
        if len(d) == 0:
            d = [np.object_]
        return d

    def _compile_ufuncs(cls):
        super()._compile_ufuncs()

        # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835552639.
        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["negative"] = np.positive
        cls._ufuncs["subtract"] = np.bitwise_xor

    ###############################################################################
    # Compile general-purpose calculate functions
    ###############################################################################

    def _compile_multiply_calculate(cls):
        global ORDER, IRREDUCIBLE_POLY
        ORDER = cls.order
        IRREDUCIBLE_POLY = cls.irreducible_poly.integer  # pylint: disable=no-member
        kwargs = {"nopython": True, "target": cls.ufunc_target} if cls.ufunc_target != "cuda" else {"target": cls.ufunc_target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)

    def _compile_reciprocal_calculate(cls):
        global ORDER, MULTIPLY_UFUNC
        ORDER = cls.order
        MULTIPLY_UFUNC = cls._ufunc_multiply()
        kwargs = {"nopython": True, "target": cls.ufunc_target} if cls.ufunc_target != "cuda" else {"target": cls.ufunc_target}
        return numba.vectorize(["int64(int64)"], **kwargs)(_reciprocal_calculate)

    def _compile_divide_calculate(cls):
        global MULTIPLY_UFUNC, RECIPROCAL_UFUNC
        MULTIPLY_UFUNC = cls._ufunc_multiply()
        RECIPROCAL_UFUNC = cls._ufunc_reciprocal()
        kwargs = {"nopython": True, "target": cls.ufunc_target} if cls.ufunc_target != "cuda" else {"target": cls.ufunc_target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_calculate)

    def _compile_power_calculate(cls):
        global MULTIPLY_UFUNC, RECIPROCAL_UFUNC
        MULTIPLY_UFUNC = cls._ufunc_multiply()
        RECIPROCAL_UFUNC = cls._ufunc_reciprocal()
        kwargs = {"nopython": True, "target": cls.ufunc_target} if cls.ufunc_target != "cuda" else {"target": cls.ufunc_target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)

    def _compile_log_calculate(cls):
        global ORDER, PRIMITIVE_ELEMENT, MULTIPLY_UFUNC
        ORDER = cls.order
        PRIMITIVE_ELEMENT = int(cls.primitive_element)
        MULTIPLY_UFUNC = cls._ufunc_multiply()
        kwargs = {"nopython": True, "target": cls.ufunc_target} if cls.ufunc_target != "cuda" else {"target": cls.ufunc_target}
        return numba.vectorize(["int64(int64)"], **kwargs)(_log_calculate)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
        return a ^ b

    def _negative_python(cls, a):
        return a

    def _subtract_python(cls, a, b):
        return a ^ b

    def _multiply_python(cls, a, b):
        """
        a in GF(2^m), can be represented as a degree m-1 polynomial in GF(2)[x]
        b in GF(2^m), can be represented as a degree m-1 polynomial in GF(2)[x]
        p(x) in GF(2)[x] with degree m is the primitive polynomial of GF(2^m)

        a * b = c
            = (a(x) * b(x)) % p(x), in GF(2)
            = c(x)
            = c
        """
        # Re-order operands such that a > b so the while loop has less loops
        if b > a:
            a, b = b, a

        c = 0
        while b > 0:
            if b & 0b1:
                c ^= a  # Add a(x) to c(x)

            b >>= 1  # Divide b(x) by x
            a <<= 1  # Multiply a(x) by x
            if a >= cls._order:
                a ^= cls._irreducible_poly_int  # Compute a(x) % p(x)

        return c

    def _reciprocal_python(cls, a):
        """
        TODO: Replace this with more efficient algorithm

        From Fermat's Little Theorem:
        a^(p^m - 1) = 1 (mod p^m), for a in GF(p^m)

        a * a^-1 = 1
        a * a^-1 = a^(p^m - 1)
            a^-1 = a^(p^m - 2)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        power = cls.order - 2
        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while power > 1:
            if power % 2 == 0:
                result_s = cls._multiply_python(result_s, result_s)
                power //= 2
            else:
                result_m = cls._multiply_python(result_m, result_s)
                power -= 1

        result = cls._multiply_python(result_m, result_s)

        return result


###############################################################################
# Galois field arithmetic, explicitly calculated without lookup tables
###############################################################################

def _multiply_calculate(a, b):  # pragma: no cover
    """
    a in GF(2^m), can be represented as a degree m-1 polynomial in GF(2)[x]
    b in GF(2^m), can be represented as a degree m-1 polynomial in GF(2)[x]
    p(x) in GF(2)[x] with degree m is the primitive polynomial of GF(2^m)

    a * b = c
          = (a(x) * b(x)) % p(x), in GF(2)
          = c(x)
          = c
    """
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


def _reciprocal_calculate(a):  # pragma: no cover
    """
    TODO: Replace this with more efficient algorithm

    From Fermat's Little Theorem:
    a^(p^m - 1) = 1 (mod p^m), for a in GF(p^m)

    a * a^-1 = 1
    a * a^-1 = a^(p^m - 1)
        a^-1 = a^(p^m - 2)
    """
    if a == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    power = ORDER - 2
    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = MULTIPLY_UFUNC(result_s, result_s)
            power //= 2
        else:
            result_m = MULTIPLY_UFUNC(result_m, result_s)
            power -= 1

    result = MULTIPLY_UFUNC(result_m, result_s)

    return result


def _divide_calculate(a, b):  # pragma: no cover
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:
        return 0
    else:
        b_inv = RECIPROCAL_UFUNC(b)
        return MULTIPLY_UFUNC(a, b_inv)


def _power_calculate(a, power):  # pragma: no cover
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
    if a == 0 and power < 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if power == 0:
        return 1
    elif power < 0:
        a = RECIPROCAL_UFUNC(a)
        power = abs(power)

    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = MULTIPLY_UFUNC(result_s, result_s)
            power //= 2
        else:
            result_m = MULTIPLY_UFUNC(result_m, result_s)
            power -= 1

    result = MULTIPLY_UFUNC(result_m, result_s)

    return result


def _log_calculate(beta):  # pragma: no cover
    """
    TODO: Replace this with more efficient algorithm

    alpha in GF(p^m) and generates field
    beta in GF(p^m)

    gamma = log_primitive_element(beta), such that: alpha^gamma = beta
    """
    if beta == 0:
        raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

    # Naive algorithm
    result = 1
    for i in range(0, ORDER-1):
        if result == beta:
            break
        result = MULTIPLY_UFUNC(result, PRIMITIVE_ELEMENT)

    return i
