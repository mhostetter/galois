"""
A module that contains a metaclass mixin that provides generic Galois field arithmetic using lookup tables.
"""
import numba
from numba import int64
import numpy as np

from ._calculate import CalculateMeta


class LookupMeta(CalculateMeta):
    """
    A mixin class that provides Galois field arithmetic using lookup tables.
    """
    # pylint: disable=no-value-for-parameter,abstract-method,unused-argument

    # Function signatures for JIT-compiled "lookup" arithmetic functions
    _UNARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64[:], int64[:], int64[:], int64))
    _BINARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64, int64[:], int64[:], int64[:], int64))

    _FUNC_CACHE_LOOKUP = {}
    _UFUNC_CACHE_LOOKUP = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._EXP = np.array([], dtype=np.int64)
        cls._LOG = np.array([], dtype=np.int64)
        cls._ZECH_LOG = np.array([], dtype=np.int64)
        cls._ZECH_E = 0

    def _build_lookup_tables(cls):
        """
        Construct EXP, LOG, and ZECH_LOG lookup tables to be used in the "lookup" arithmetic functions
        """
        # Only construct the LUTs for this field once
        if cls._EXP.size > 0:
            return

        order = cls.order
        primitive_element = int(cls.primitive_element)
        add = lambda a, b: cls._func_python("add")(a, b, cls.characteristic, cls.degree, cls._irreducible_poly_int)
        multiply = lambda a, b: cls._func_python("multiply")(a, b, cls.characteristic, cls.degree, cls._irreducible_poly_int)

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
            element = multiply(element, primitive_element)
            cls._EXP[i] = element

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, order):
            one_plus_element = add(1, cls._EXP[i])
            cls._ZECH_LOG[i] = cls._LOG[one_plus_element]

        if not cls._EXP[order - 1] == 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not cyclic with size {order - 1}, which means the primitive element {cls.primitive_element} does not have multiplicative order {order - 1} and therefore isn't a multiplicative generator for {cls.name}.")
        if not len(set(cls._EXP[0:order - 1])) == order - 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not unique, which means the primitive element {cls.primitive_element} has order less than {order - 1} and is not a multiplicative generator of {cls.name}.")
        if not len(set(cls._LOG[1:order])) == order - 1:
            raise RuntimeError(f"The log lookup table for {cls.name} is not unique.")

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[order:2*order] = cls._EXP[1:1 + order]

    def _func_lookup(cls, name):  # pylint: disable=no-self-use
        """
        Returns an arithmetic function using lookup tables. These functions are once-compiled and shared for all Galois fields. The only difference
        between Galois fields are the lookup tables that are passed in as inputs.
        """
        key = (name,)

        if key not in cls._FUNC_CACHE_LOOKUP:
            function = getattr(cls, f"_{name}_lookup")
            if cls._UFUNC_TYPE[name] == "unary":
                cls._FUNC_CACHE_LOOKUP[key] = numba.jit("int64(int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)
            else:
                cls._FUNC_CACHE_LOOKUP[key] = numba.jit("int64(int64, int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)

        return cls._FUNC_CACHE_LOOKUP[key]

    def _ufunc_lookup(cls, name):
        """
        Returns an arithmetic ufunc using lookup tables. These ufuncs are compiled for each Galois field since the lookup tables are compiled
        into the ufuncs as constants.
        """
        key = (name, cls.characteristic, cls.degree, cls._irreducible_poly_int)

        if key not in cls._UFUNC_CACHE_LOOKUP:
            EXP = cls._EXP
            LOG = cls._LOG
            ZECH_LOG = cls._ZECH_LOG
            ZECH_E = cls._ZECH_E

            function = getattr(cls, f"_{name}_lookup")
            if cls._UFUNC_TYPE[name] == "unary":
                cls._UFUNC_CACHE_LOOKUP[key] = numba.vectorize(["int64(int64)"], nopython=True)(lambda a: function(a, EXP, LOG, ZECH_LOG, ZECH_E))
            else:
                cls._UFUNC_CACHE_LOOKUP[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(lambda a, b: function(a, b, EXP, LOG, ZECH_LOG, ZECH_E))

        return cls._UFUNC_CACHE_LOOKUP[key]

    ###############################################################################
    # Arithmetic functions using lookup tables
    ###############################################################################

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _add_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a + b = α^m + α^n
            = α^m * (1 + α^(n - m))  # If n is larger, factor out α^m
            = α^m * α^ZECH_LOG(n - m)
            = α^(m + ZECH_LOG(n - m))
        """
        if a == 0:
            return b
        elif b == 0:
            return a

        m = LOG[a]
        n = LOG[b]

        if m > n:
            # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If m is
            # larger than n, switch a and b in the addition.
            m, n = n, m

        if n - m == ZECH_E:
            # zech_log(zech_e) = -Inf and α^(-Inf) = 0
            return 0
        else:
            return EXP[m + ZECH_LOG[n - m]]

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _negative_lookup(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        -a = -α^m
        = -1 * α^m
        = α^e * α^m
        = α^(e + m)
        """
        if a == 0:
            return 0
        else:
            m = LOG[a]
            return EXP[ZECH_E + m]

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _subtract_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a - b = α^m - α^n
            = α^m + (-α^n)
            = α^m + (-1 * α^n)
            = α^m + (α^e * α^n)
            = α^m + α^(e + n)
        """
        ORDER = LOG.size

        # Same as addition if n = log(b) + e
        m = LOG[a]
        n = LOG[b] + ZECH_E

        if b == 0:
            return a
        elif a == 0:
            return EXP[n]

        if m > n:
            # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If m is
            # larger than n, switch a and b in the addition.
            m, n = n, m

        z = n - m
        if z == ZECH_E:
            # zech_log(zech_e) = -Inf and α^(-Inf) = 0
            return 0
        if z >= ORDER - 1:
            # Reduce index of ZECH_LOG by the multiplicative order of the field `ORDER - 1`
            z -= ORDER - 1

        return EXP[m + ZECH_LOG[z]]

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _multiply_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a * b = α^m * α^n
            = α^(m + n)
        """
        if a == 0 or b == 0:
            return 0
        else:
            m = LOG[a]
            n = LOG[b]
            return EXP[m + n]

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _reciprocal_lookup(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

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

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _divide_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a / b = α^m / α^n
            = α^(m - n)
            = 1 * α^(m - n)
            = α^(ORDER - 1) * α^(m - n)
            = α^(ORDER - 1 + m - n)
        """
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            ORDER = LOG.size
            m = LOG[a]
            n = LOG[b]
            return EXP[(ORDER - 1) + m - n]  # We add `ORDER - 1` to guarantee the index is non-negative

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _power_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b in Z

        a ** b = α^m ** b
            = α^(m * b)
            = α^(m * ((b // (ORDER - 1))*(ORDER - 1) + b % (ORDER - 1)))
            = α^(m * ((b // (ORDER - 1))*(ORDER - 1)) * α^(m * (b % (ORDER - 1)))
            = 1 * α^(m * (b % (ORDER - 1)))
            = α^(m * (b % (ORDER - 1)))
        """
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif a == 0:
            return 0
        else:
            ORDER = LOG.size
            m = LOG[a]
            return EXP[(m * b) % (ORDER - 1)]  # TODO: Do b % (ORDER - 1) first? b could be very large and overflow int64

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _log_lookup(beta, alpha, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        log(beta, α) = log(α^m, α)
                    = m
        """
        if beta == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        return LOG[beta]
