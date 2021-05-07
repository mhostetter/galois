import numba
import numpy as np

from .dtypes import DTYPES
from .meta_gf import GFMeta
from .poly import Poly

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
DEGREE = None  # The prime power `m` of the Galois field
ORDER = None  # The field's order `p^m`
PRIMITIVE_ELEMENT = None  # The field's primitive element
IRREDUCIBLE_POLY_INT = None  # The field's primitive polynomial in decimal form

# Placeholder functions to be replaced by JIT-compiled function
MULTIPLY_JIT = lambda x, y: x * y
MULTIPLICATIVE_INVERSE_JIT = lambda x: 1 / x


class GF2mMeta(GFMeta):
    """
    An abstract base class for all :math:`\\mathrm{GF}(2^m)` field array classes.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly_int = cls._irreducible_poly.integer  # pylint: disable=no-member
        cls._primitive_element_int = cls._primitive_element.integer
        cls._prime_subfield = kwargs["prime_subfield"]

        cls.compile(kwargs["mode"], kwargs["target"])

        # Convert primitive element from a poly in GF(2)[x] to an integer
        cls._primitive_element = cls(cls._primitive_element.integer)

        # Determine if the irreducible polynomial is primitive
        cls._is_primitive_poly = cls._irreducible_poly(cls.primitive_element, field=cls) == 0

    @property
    def dtypes(cls):
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1]
        if len(d) == 0:
            d = [np.object_]
        return d

    def _compile_jit_calculate(cls, target):
        global CHARACTERISTIC, ORDER, PRIMITIVE_ELEMENT, IRREDUCIBLE_POLY_INT, MULTIPLY_JIT, MULTIPLICATIVE_INVERSE_JIT
        CHARACTERISTIC = cls.characteristic
        ORDER = cls.order
        if isinstance(cls._primitive_element, Poly):
            PRIMITIVE_ELEMENT = cls._primitive_element.integer
        else:
            PRIMITIVE_ELEMENT = int(cls._primitive_element)
        IRREDUCIBLE_POLY_INT = cls.irreducible_poly.integer  # pylint: disable=no-member

        # JIT-compile add,  multiply, and multiplicative inverse routines for reference in polynomial evaluation routine
        MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(_multiply_calculate)
        MULTIPLICATIVE_INVERSE_JIT = numba.jit("int64(int64)", nopython=True)(_multiplicative_inverse_calculate)

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # Create numba JIT-compiled ufuncs
        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["subtract"] = np.bitwise_xor
        cls._ufuncs["multiply"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
        cls._ufuncs["divide"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_calculate)
        cls._ufuncs["negative"] = numba.vectorize(["int64(int64)"], **kwargs)(_additive_inverse_calculate)
        cls._ufuncs["reciprocal"] = numba.vectorize(["int64(int64)"], **kwargs)(_multiplicative_inverse_calculate)
        cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
        cls._ufuncs["log"] = numba.vectorize(["int64(int64)"], **kwargs)(_log_calculate)
        cls._ufuncs["poly_eval"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_calculate)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
        return a ^ b

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

    def _additive_inverse_python(cls, a):
        return a

    def _multiplicative_inverse_python(cls, a):
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
            a ^= IRREDUCIBLE_POLY_INT  # Compute a(x) % p(x)

    return c


def _divide_calculate(a, b):  # pragma: no cover
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:
        return 0
    else:
        b_inv = MULTIPLICATIVE_INVERSE_JIT(b)
        return MULTIPLY_JIT(a, b_inv)


def _additive_inverse_calculate(a):  # pragma: no cover
    return a


def _multiplicative_inverse_calculate(a):  # pragma: no cover
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
            result_s = MULTIPLY_JIT(result_s, result_s)
            power //= 2
        else:
            result_m = MULTIPLY_JIT(result_m, result_s)
            power -= 1

    result = MULTIPLY_JIT(result_m, result_s)

    return result


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
        a = MULTIPLICATIVE_INVERSE_JIT(a)
        power = abs(power)

    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = MULTIPLY_JIT(result_s, result_s)
            power //= 2
        else:
            result_m = MULTIPLY_JIT(result_m, result_s)
            power -= 1

    result = MULTIPLY_JIT(result_m, result_s)

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
        result = MULTIPLY_JIT(result, PRIMITIVE_ELEMENT)

    return i


def _poly_eval_calculate(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = coeffs[j] ^ MULTIPLY_JIT(results[i], values[i])
