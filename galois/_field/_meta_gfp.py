import numba
import numpy as np

from .._modular import primitive_root

from ._dtypes import DTYPES
from ._meta_class import FieldClass, DirMeta
from ._meta_ufunc import  _FUNCTION_TYPE
from ._poly import Poly


class GFpMeta(FieldClass, DirMeta):
    """
    An metaclass for all GF(p) classes.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._prime_subfield = cls

        cls.compile(kwargs["compile"])

    @property
    def irreducible_poly(cls):
        if cls._irreducible_poly is None:
            cls._irreducible_poly = Poly([1, -cls.primitive_element], field=cls)
        return cls._irreducible_poly

    @property
    def primitive_element(cls):
        if cls._primitive_element is None:
            cls._primitive_element = primitive_root(cls._characteristic)
        return cls(cls._primitive_element)

    @property
    def dtypes(cls):
        max_dtype = DTYPES[-1]
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(d) == 0:
            d = [np.object_]
        return d

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

    def _ufunc(cls, name):
        # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835548405.
        if name not in cls._ufuncs and cls.ufunc_mode == "jit-lookup" and name in ["add", "negative", "subtract"]:
            cls._ufuncs[name] = cls._calculate_ufunc(name)
        return super()._ufunc(name)

    def _calculate_ufunc(cls, name):
        return compile_ufunc(name, cls.characteristic, cls.degree, cls._irreducible_poly_int)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
        c = a + b
        if c >= cls.order:
            c -= cls.order
        return c

    def _negative_python(cls, a):
        if a == 0:
            return 0
        else:
            return cls.order - a

    def _subtract_python(cls, a, b):
        if a >= b:
            return a - b
        else:
            return cls.order + a - b

    def _multiply_python(cls, a, b):
        return (a * b) % cls.order

    def _reciprocal_python(cls, a):
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        r2, r1 = cls.order, a
        t2, t1 = 0, 1

        while r1 != 0:
            q = r2 // r1
            r2, r1 = r1, r2 - q*r1
            t2, t1 = t1, t2 - q*t1

        if t2 < 0:
            t2 += cls.order

        return t2


###############################################################################
# Compile functions
###############################################################################

CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
DEGREE = None  # The prime power `m` of the Galois field
IRREDUCIBLE_POLY = None  # The field's primitive polynomial in integer form

RECIPROCAL = lambda a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY: 1 / a

# pylint: disable=redefined-outer-name,unused-argument


def compile_jit(name, reset=True):
    """
    Compile a JIT arithmetic function. These can be cached.
    """
    if name not in compile_jit.cache:
        global RECIPROCAL

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
        global CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY, RECIPROCAL
        CHARACTERISTIC = CHARACTERISTIC_
        DEGREE = DEGREE_
        IRREDUCIBLE_POLY = IRREDUCIBLE_POLY_

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

@numba.extending.register_jitable(inline="always")
def add(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    c = a + b
    if c >= CHARACTERISTIC:
        c -= CHARACTERISTIC
    return c


def add_ufunc(a, b):  # pragma: no cover
    return add(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def negative(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if a == 0:
        return 0
    else:
        return CHARACTERISTIC - a


def negative_ufunc(a):  # pragma: no cover
    return negative(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def subtract(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if a >= b:
        return a - b
    else:
        return CHARACTERISTIC + a - b


def subtract_ufunc(a, b):  # pragma: no cover
    return subtract(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def multiply(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return (a * b) % CHARACTERISTIC


def multiply_ufunc(a, b):  # pragma: no cover
    return multiply(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def reciprocal(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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
        return (a * b_inv) % CHARACTERISTIC


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
            result_s = (result_s * result_s) % CHARACTERISTIC
            b //= 2
        else:
            result_m = (result_m * result_s) % CHARACTERISTIC
            b -= 1

    result = (result_m * result_s) % CHARACTERISTIC

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
        result = (result * b) % CHARACTERISTIC

    return i


def log_ufunc(a, b):  # pragma: no cover
    return log(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


def reset_globals():
    """
    Reset the global variable so when the pure-python ufuncs call these routines, they reference
    the correct pure-python functions (not JIT functions or JIT-compiled ufuncs).
    """
    global RECIPROCAL
    RECIPROCAL = reciprocal

reset_globals()
