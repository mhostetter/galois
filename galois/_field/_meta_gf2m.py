import numba
import numpy as np

from ._meta_class import FieldClass, DirMeta
from ._meta_ufunc import  _FUNCTION_TYPE


class GF2mMeta(FieldClass, DirMeta):
    """
    A metaclass for all GF(2^m) classes.
    """
    # pylint: disable=abstract-method,no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._prime_subfield = kwargs["prime_subfield"]

        cls.compile(kwargs["compile"])

        # Determine if the irreducible polynomial is primitive
        if cls._is_primitive_poly is None:
            cls._is_primitive_poly = cls._poly_evaluate_python(cls._irreducible_poly.coeffs, cls.primitive_element) == 0

    def _compile_ufuncs(cls):
        super()._compile_ufuncs()

        # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835552639.
        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["negative"] = np.positive
        cls._ufuncs["subtract"] = np.bitwise_xor

    ###############################################################################
    # Individual JIT arithmetic functions, pre-compiled (cached)
    ###############################################################################

    def _calculate_jit(cls, name):
        return compile_jit(name)

    def _python_func(cls, name):
        return eval(f"{name}")

    ###############################################################################
    # Individual ufuncs, compiled on-demand
    ###############################################################################

    def _calculate_ufunc(cls, name):
        return compile_ufunc(name, cls.characteristic, cls.degree, cls._irreducible_poly_int)

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
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        exponent = cls.order - 2
        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while exponent > 1:
            if exponent % 2 == 0:
                result_s = cls._multiply_python(result_s, result_s)
                exponent //= 2
            else:
                result_m = cls._multiply_python(result_m, result_s)
                exponent -= 1

        result = cls._multiply_python(result_m, result_s)

        return result


###############################################################################
# Compile functions
###############################################################################

CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
DEGREE = None  # The prime power `m` of the Galois field
IRREDUCIBLE_POLY = None  # The field's primitive polynomial in integer form

MULTIPLY = lambda a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY: a * b
RECIPROCAL = lambda a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY: 1 / a

# pylint: disable=redefined-outer-name,unused-argument


def compile_jit(name, reset=True):
    """
    Compile a JIT arithmetic function. These can be cached.
    """
    if name not in compile_jit.cache:
        global MULTIPLY, RECIPROCAL

        if name in ["reciprocal", "divide", "power", "log"]:
            MULTIPLY = compile_jit("multiply", reset=False)
        if name in ["divide", "power"]:
            RECIPROCAL = compile_jit("reciprocal", reset=False)

        function = eval(f"{name}")
        if _FUNCTION_TYPE[name] == "unary":
            compile_jit.cache[name] = numba.jit(["int64(int64, int64, int64, int64)"], nopython=True, cache=True)(function)
        else:
            compile_jit.cache[name] = numba.jit(["int64(int64, int64, int64, int64, int64)"], nopython=True, cache=True)(function)

        if reset:
            reset_globals()

    return compile_jit.cache[name]

compile_jit.cache = {}


def compile_ufunc(name, CHARACTERISTIC_, DEGREE_, IRREDUCIBLE_POLY_):
    """
    Compile an arithmetic ufunc. These cannot be cached as the field parameters are compiled into the binary.
    """
    key = (name, CHARACTERISTIC_, DEGREE_, IRREDUCIBLE_POLY_)
    if key not in compile_ufunc.cache:
        global CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY, MULTIPLY, RECIPROCAL
        CHARACTERISTIC = CHARACTERISTIC_
        DEGREE = DEGREE_
        IRREDUCIBLE_POLY = IRREDUCIBLE_POLY_

        if name in ["reciprocal", "divide", "power", "log"]:
            MULTIPLY = compile_jit("multiply", reset=False)
        if name in ["divide", "power"]:
            RECIPROCAL = compile_jit("reciprocal", reset=False)

        function = eval(f"{name}_ufunc")
        if _FUNCTION_TYPE[name] == "unary":
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64)"], nopython=True)(function)
        else:
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(function)

        reset_globals()

    return compile_ufunc.cache[key]

compile_ufunc.cache = {}


###############################################################################
# Arithmetic explicitly calculated
###############################################################################

def add(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return a ^ b


def negative(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return a


def subtract(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return a ^ b


@numba.extending.register_jitable(inline="always")
def multiply(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    """
    a in GF(2^m), can be represented as a degree m-1 polynomial in GF(2)[x]
    b in GF(2^m), can be represented as a degree m-1 polynomial in GF(2)[x]
    p(x) in GF(2)[x] with degree m is the primitive polynomial of GF(2^m)

    a * b = c
          = (a(x) * b(x)) % p(x), in GF(2)
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


def multiply_ufunc(a, b):  # pragma: no cover
    return multiply(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def reciprocal(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    """
    From Fermat's Little Theorem:
    a^(p^m - 1) = 1 (mod p^m), for a in GF(p^m)

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


def reciprocal_ufunc(a):  # pragma: no cover
    return reciprocal(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def divide(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:
        return 0
    else:
        b_inv = RECIPROCAL(b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
        return MULTIPLY(a, b_inv, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


def divide_ufunc(a, b):  # pragma: no cover
    return divide(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def power(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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


def power_ufunc(a, b):  # pragma: no cover
    return power(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def log(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    """
    TODO: Replace this with more efficient algorithm

    a in GF(p^m)
    b in GF(p^m) and generates field

    c = log(a, b), such that b^a = c
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


def log_ufunc(a, b):  # pragma: no cover
    return log(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


def reset_globals():
    """
    Reset the global variable so when the pure-python ufuncs call these routines, they reference
    the correct pure-python functions (not JIT functions or JIT-compiled ufuncs).
    """
    global MULTIPLY, RECIPROCAL
    MULTIPLY = multiply
    RECIPROCAL = reciprocal

reset_globals()
