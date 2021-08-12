"""
A module that contains a metaclass mixin that provides generic Galois field arithmetic using lookup tables.
"""
import numba
from numba import int64
import numpy as np

from ._properties import PropertiesMeta


class LookupMeta(PropertiesMeta):
    """
    A mixin class that provides Galois field arithmetic using lookup tables.
    """
    # pylint: disable=no-value-for-parameter

    _UNARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64[:], int64[:], int64[:], int64))
    _BINARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64, int64[:], int64[:], int64[:], int64))

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

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._EXP = np.array([], dtype=np.int64)
        cls._LOG = np.array([], dtype=np.int64)
        cls._ZECH_LOG = np.array([], dtype=np.int64)
        cls._ZECH_E = 0

    def _compile_ufuncs(cls):
        cls._ufuncs = {}  # Reset the dictionary so each ufunc will get recompiled
        if cls.ufunc_mode == "jit-lookup":
            cls._build_lookup_tables()

    def _build_lookup_tables(cls):
        if cls._EXP.size > 0:
            return

        order = cls.order
        primitive_element = int(cls.primitive_element)

        cls._EXP = np.zeros(2*order, dtype=np.int64)
        cls._LOG = np.zeros(order, dtype=np.int64)
        cls._ZECH_LOG = np.zeros(order, dtype=np.int64)
        if cls.characteristic == 2:
            cls._ZECH_E = 0
        else:
            cls._ZECH_E = (cls.order - 1) // 2

        element = 1
        cls._EXP[0] = element
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, order):
            # Increment by multiplying by the primitive element, which is a multiplicative generator of the field
            element = cls._multiply_python(element, primitive_element)
            cls._EXP[i] = element

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, order):
            one_plus_element = cls._add_python(1, cls._EXP[i])
            cls._ZECH_LOG[i] = cls._LOG[one_plus_element]

        if not cls._EXP[order - 1] == 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not cyclic with size {order - 1}, which means the primitive element {cls.primitive_element} does not have multiplicative order {order - 1} and therefore isn't a multiplicative generator for {cls.name}.")
        if not len(set(cls._EXP[0:order - 1])) == order - 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not unique, which means the primitive element {cls.primitive_element} has order less than {order - 1} and is not a multiplicative generator of {cls.name}.")
        if not len(set(cls._LOG[1:order])) == order - 1:
            raise RuntimeError(f"The log lookup table for {cls.name} is not unique.")

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[order:2*order] = cls._EXP[1:1 + order]

    ###############################################################################
    # Individual JIT arithmetic functions, pre-compiled (cached)
    ###############################################################################

    def _lookup_jit(cls, name):  # pylint: disable=no-self-use
        return compile_jit(name)

    ###############################################################################
    # Individual ufuncs, compiled on-demand
    ###############################################################################

    def _lookup_ufunc(cls, name):
        return compile_ufunc(name, cls.characteristic, cls.degree, cls._irreducible_poly_int, cls._EXP, cls._LOG, cls._ZECH_LOG, cls._ZECH_E)


###############################################################################
# Compile arithmetic functions using lookup tables
###############################################################################

EXP = []  # EXP[i] = α^i
LOG = []  # LOG[i] = x, such that α^x = i
ZECH_LOG = []  # ZECH_LOG[i] = log(1 + α^i)
ZECH_E = 0  # α^ZECH_E = -1, ZECH_LOG[ZECH_E] = -Inf

# pylint: disable=redefined-outer-name,unused-argument


def compile_jit(name):
    """
    Compile a JIT arithmetic function. These can be cached.
    """
    if name not in compile_jit.cache:
        function = eval(f"{name}")
        if LookupMeta._UFUNC_TYPE[name] == "unary":
            compile_jit.cache[name] = numba.jit("int64(int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)
        else:
            compile_jit.cache[name] = numba.jit("int64(int64, int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)

    return compile_jit.cache[name]

compile_jit.cache = {}


def compile_ufunc(name, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY, EXP_, LOG_, ZECH_LOG_, ZECH_E_):
    """
    Compile an arithmetic ufunc. These cannot be cached as the lookup tables are compiled into the binary.
    """
    key = (name, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
    if key not in compile_ufunc.cache:
        global EXP, LOG, ZECH_LOG, ZECH_E
        EXP = EXP_
        LOG = LOG_
        ZECH_LOG = ZECH_LOG_
        ZECH_E = ZECH_E_

        function = eval(f"{name}_ufunc")
        if LookupMeta._UFUNC_TYPE[name] == "unary":
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64)"], nopython=True)(function)
        else:
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(function)

    return compile_ufunc.cache[key]

compile_ufunc.cache = {}


###############################################################################
# Arithmetic using lookup tables
###############################################################################

@numba.extending.register_jitable(inline="always")
def add(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a + b = α^m + α^n
          = α^m * (1 + α^(n - m))  # If n is larger, factor out α^m
          = α^m * α^ZECH_LOG(n - m)
          = α^(m + ZECH_LOG(n - m))
    """
    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return b
    elif b == 0:
        return a

    m = LOG[a]
    n = LOG[b]

    if m > n:
        # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If
        # m is larger than n, switch a and b in the addition.
        m, n = n, m

    if n - m == ZECH_E:
        # ZECH_LOG[ZECH_E] = -Inf and α^(-Inf) = 0
        return 0
    else:
        return EXP[m + ZECH_LOG[n - m]]


def add_ufunc(a, b):  # pragma: no cover
    return add(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def negative(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    -a = -α^n
       = -1 * α^n
       = α^e * α^n
       = α^(e + n)
    """
    if a == 0:  # LOG[0] = -Inf, so catch this condition
        return 0
    else:
        n = LOG[a]
        return EXP[ZECH_E + n]


def negative_ufunc(a):  # pragma: no cover
    return negative(a, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def subtract(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a - b = α^m - α^n
          = α^m + (-α^n)
          = α^m + (-1 * α^n)
          = α^m + (α^e * α^n)
          = α^m + α^(e + n)
    """
    ORDER = LOG.size

    # Same as addition if n = LOG[b] + e
    m = LOG[a]
    n = LOG[b] + ZECH_E

    # LOG[0] = -Inf, so catch these conditions
    if b == 0:
        return a
    elif a == 0:
        return EXP[n]

    if m > n:
        # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If
        # m is larger than n, switch a and b in the addition.
        m, n = n, m

    z = n - m
    if z == ZECH_E:
        # ZECH_LOG[ZECH_E] = -Inf and α^(-Inf) = 0
        return 0
    if z >= ORDER - 1:
        # Reduce index of ZECH_LOG by the multiplicative order of the field, i.e. `order - 1`
        z -= ORDER - 1

    return EXP[m + ZECH_LOG[z]]


def subtract_ufunc(a, b):  # pragma: no cover
    return subtract(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def multiply(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a * b = α^m * α^n
          = α^(m + n)
    """
    if a == 0 or b == 0:  # LOG[0] = -Inf, so catch these conditions
        return 0
    else:
        m = LOG[a]
        n = LOG[b]
        return EXP[m + n]


def multiply_ufunc(a, b):  # pragma: no cover
    return multiply(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def reciprocal(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    1 / a = 1 / α^m
          = α^(-m)
          = 1 * α^(-m)
          = α^(ORDER - 1) * α^(-m)
          = α^(ORDER - 1 - m)
    """
    if a == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    ORDER = LOG.size
    m = LOG[a]
    return EXP[(ORDER - 1) - m]


def reciprocal_ufunc(a):  # pragma: no cover
    return reciprocal(a, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def divide(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a / b = α^m / α^n
          = α^(m - n)
          = 1 * α^(m - n)
          = α^(ORDER - 1) * α^(m - n)
          = α^(ORDER - 1 + m - n)
    """
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:  # LOG[0] = -Inf, so catch this condition
        return 0
    else:
        ORDER = LOG.size
        m = LOG[a]
        n = LOG[b]
        return EXP[(ORDER - 1) + m - n]  # We add `ORDER - 1` to guarantee the index is non-negative


def divide_ufunc(a, b):  # pragma: no cover
    return divide(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def power(a, b_int, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    b_int in Z
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a ** b_int = α^m ** b_int
               = α^(m * b_int)
               = α^(m * ((b_int // (ORDER - 1))*(ORDER - 1) + b_int % (ORDER - 1)))
               = α^(m * ((b_int // (ORDER - 1))*(ORDER - 1)) * α^(m * (b_int % (ORDER - 1)))
               = 1 * α^(m * (b_int % (ORDER - 1)))
               = α^(m * (b_int % (ORDER - 1)))
    """
    if a == 0 and b_int < 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if b_int == 0:
        return 1
    elif a == 0:  # LOG[0] = -Inf, so catch this condition
        return 0
    else:
        ORDER = LOG.size
        m = LOG[a]
        return EXP[(m * b_int) % (ORDER - 1)]


def power_ufunc(a, b_int):  # pragma: no cover
    return power(a, b_int, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def log(beta, alpha, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    log(beta, α) = log(α^m, α)
                 = m
    """
    if beta == 0:
        raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

    return LOG[beta]


def log_ufunc(beta, alpha):  # pragma: no cover
    return log(beta, alpha, EXP, LOG, ZECH_LOG, ZECH_E)
