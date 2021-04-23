import numba
import numpy as np

# Placeholder globals that will be set in _compile_jit_lookup()
CHARACTERISTIC = None  # The field's prime characteristic `p`
ORDER = None  # The field's order `p^m`

EXP = []  # EXP[i] = α^i
LOG = []  # LOG[i] = x, such that α^x = i
ZECH_LOG = []  # ZECH_LOG[i] = log(1 + α^i)
ZECH_E = None  # α^ZECH_E = -1, ZECH_LOG[ZECH_E] = -Inf

ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y


class TargetMixin(type):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._EXP = None
        cls._LOG = None
        cls._ZECH_LOG = None
        cls._ufuncs = {}

        # Integer representations of the field's primitive element and primitive polynomial to be used in the
        # pure python ufunc implementations for `ufunc_mode = "python-calculate"`
        cls._primitive_element_dec = None
        cls._irreducible_poly_dec = None

    def _fill_in_lookup_tables(cls):
        """
        To be implemented in PrimeTargetMixin and ExtensionTargetMixin. Each class GF2Meta, GF2mMeta,
        GFpMeta, and GFpmMeta will inherit from either PrimeTargetMixin or ExtensionLookupMixin.
        """
        raise NotImplementedError

    def _build_lookup_tables(cls):
        order = cls.order
        dtype = np.int64
        if order > np.iinfo(dtype).max:
            raise RuntimeError(f"Cannot build lookup tables for {cls.name} since the elements cannot be represented with dtype {dtype}.")

        cls._EXP = np.zeros(2*order, dtype=dtype)
        cls._LOG = np.zeros(order, dtype=dtype)
        cls._ZECH_LOG = np.zeros(order, dtype=dtype)

        cls._fill_in_lookup_tables()

        if not cls._EXP[order - 1] == 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not cyclic with size {order - 1}, which means the primitive element {cls.primitive_element} does not have multiplicative order {order - 1} and therefore isn't a multiplicative generator for {cls.name}.")
        if not len(set(cls._EXP[0:order - 1])) == order - 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not unique, which means the primitive element {cls.primitive_element} has order less than {order - 1} and is not a multiplicative generator of {cls.name}.")
        if not len(set(cls._LOG[1:order])) == order - 1:
            raise RuntimeError(f"The log lookup table for {cls.name} is not unique.")

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[order:2*order] = cls._EXP[1:1 + order]

    def _compile_jit_lookup(cls, target):
        """
        A method to JIT-compile the standard lookup arithmetic for any field. The functions that are
        JIT compiled are at the bottom of this file.
        """
        global CHARACTERISTIC, ORDER, EXP, LOG, ZECH_LOG, ZECH_E, ADD_JIT, MULTIPLY_JIT

        # Build the lookup tables if they don't exist
        if cls._EXP is None:
            cls._build_lookup_tables()

        # Export lookup tables to global variables so JIT compiling can cache the tables in the binaries
        CHARACTERISTIC = cls.characteristic
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        ZECH_LOG = cls._ZECH_LOG
        if cls.characteristic == 2:
            ZECH_E = 0
        else:
            ZECH_E = (cls.order - 1) // 2

        # JIT-compile add and multiply routines for reference in other routines
        ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(_add_lookup)
        MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(_multiply_lookup)

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # TODO: Use smallest possible dtype for ufuncs
        # d = np.dtype(self.dtypes[0]).name

        # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
        cls._ufuncs["add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_lookup)
        cls._ufuncs["subtract"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_lookup)
        cls._ufuncs["multiply"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_lookup)
        cls._ufuncs["divide"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_lookup)
        cls._ufuncs["negative"] = numba.vectorize(["int64(int64)"], **kwargs)(_additive_inverse_lookup)
        cls._ufuncs["reciprocal"] = numba.vectorize(["int64(int64)"], **kwargs)(_multiplicative_inverse_lookup)
        cls._ufuncs["multiple_add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiple_add_lookup)
        cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_lookup)
        cls._ufuncs["log"] = numba.vectorize(["int64(int64)"], **kwargs)(_log_lookup)
        cls._ufuncs["poly_eval"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_lookup)

    def _compile_jit_calculate(cls, target):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta. The functions that will
        be JIT-compiled will be located at the bottom of those files.
        """
        raise NotImplementedError

    def _compile_python_calculate(cls):
        cls._ufuncs["add"] = np.frompyfunc(cls._add_python, 2, 1)
        cls._ufuncs["subtract"] = np.frompyfunc(cls._subtract_python, 2, 1)
        cls._ufuncs["multiply"] = np.frompyfunc(cls._multiply_python, 2, 1)
        cls._ufuncs["divide"] = np.frompyfunc(cls._divide_python, 2, 1)
        cls._ufuncs["negative"] = np.frompyfunc(cls._additive_inverse_python, 1, 1)
        cls._ufuncs["reciprocal"] = np.frompyfunc(cls._multiplicative_inverse_python, 1, 1)
        cls._ufuncs["multiple_add"] = np.frompyfunc(cls._multiple_add_python, 2, 1)
        cls._ufuncs["power"] = np.frompyfunc(cls._power_python, 2, 1)
        cls._ufuncs["log"] = np.frompyfunc(cls._log_python, 1, 1)
        cls._ufuncs["poly_eval"] = np.vectorize(cls._poly_eval_python, excluded=["coeffs"], otypes=[np.object_])

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
        """
        To be implemented by GF2, GF2m, GFp, and GFpm.
        """
        raise NotImplementedError

    def _subtract_python(cls, a, b):
        """
        To be implemented by GF2, GF2m, GFp, and GFpm.
        """
        raise NotImplementedError

    def _multiply_python(cls, a, b):
        """
        To be implemented by GF2, GF2m, GFp, and GFpm.
        """
        raise NotImplementedError

    def _divide_python(cls, a, b):
        if a == 0 or b == 0:
            # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
            return 0
        b_inv = cls._multiplicative_inverse_python(b)
        return cls._multiply_python(a, b_inv)

    def _additive_inverse_python(cls, a):
        """
        To be implemented by GF2, GF2m, GFp, and GFpm.
        """
        raise NotImplementedError

    def _multiplicative_inverse_python(cls, a):
        """
        To be implemented by GF2, GF2m, GFp, and GFpm.
        """
        raise NotImplementedError

    def _multiple_add_python(cls, a, multiple):
        b = multiple % cls.characteristic
        return cls._multiply_python(a, b)

    def _power_python(cls, a, power):
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
        # NOTE: The a == 0 and b < 0 condition will be caught outside of the the ufunc and raise ZeroDivisonError
        if power == 0:
            return 1
        elif power < 0:
            a = cls._multiplicative_inverse_python(a)
            power = abs(power)

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

    def _log_python(cls, beta):
        """
        TODO: Replace this with more efficient algorithm

        α in GF(p^m) and generates field
        beta in GF(p^m)

        gamma = log_primitive_element(beta), such that: α^gamma = beta
        """
        # Naive algorithm
        result = 1
        for i in range(0, cls.order - 1):
            if result == beta:
                break
            result = cls._multiply_python(result, cls.primitive_element)
        return i

    def _poly_eval_python(cls, coeffs, values):
        result = coeffs[0]
        for j in range(1, coeffs.size):
            p = cls._multiply_python(result, values)
            result = cls._add_python(coeffs[j], p)
        return result


###################################################################################
# Galois field arithmetic for any field using EXP, LOG, and ZECH_LOG lookup tables
###################################################################################

def _add_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a + b = α^m + α^n
          = α^m * (1 + α^(n - m))  # If n is larger, factor out α^m
          = α^m * α^ZECH_LOG(n - m)
          = α^(m + ZECH_LOG(n - m))
    """
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return b
    if b == 0:
        return a

    if m > n:
        # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If
        # m is larger than n, switch a and b in the addition.
        m, n = n, m

    if n - m == ZECH_E:
        # ZECH_LOG[ZECH_E] = -Inf and α^(-Inf) = 0
        return 0

    return EXP[m + ZECH_LOG[n - m]]


def _subtract_lookup(a, b):  # pragma: no cover
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
    # Same as addition if n = LOG[b] + e
    m = LOG[a]
    n = LOG[b] + ZECH_E

    # LOG[0] = -Inf, so catch these conditions
    if b == 0:
        return a
    if a == 0:
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


def _multiply_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a * b = α^m * α^n
          = α^(m + n)
    """
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0 or b == 0:
        return 0

    return EXP[m + n]


def _divide_lookup(a, b):  # pragma: no cover
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
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0 or b == 0:
        # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0

    # We add `ORDER - 1` to guarantee the index is non-negative
    return EXP[(ORDER - 1) + m - n]


def _additive_inverse_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    -a = -α^n
       = -1 * α^n
       = α^e * α^n
       = α^(e + n)
    """
    n = LOG[a]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return 0

    return EXP[ZECH_E + n]


def _multiplicative_inverse_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    1 / a = 1 / α^m
          = α^(-m)
          = 1 * α^(-m)
          = α^(ORDER - 1) * α^(-m)
          = α^(ORDER - 1 - m)
    """
    m = LOG[a]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        # NOTE: The a == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0

    return EXP[(ORDER - 1) - m]


def _multiple_add_lookup(a, b_int):  # pragma: no cover
    """
    a in GF(p^m)
    b_int in Z
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}
    b in GF(p^m)

    a . b_int = a + a + ... + a = b_int additions of a
    a . p_int = 0, where p_int is the prime characteristic of the field

    a . b_int = a * ((b_int // p_int)*p_int + b_int % p_int)
              = a * ((b_int // p_int)*p_int) + a * (b_int % p_int)
              = 0 + a * (b_int % p_int)
              = a * (b_int % p_int)
              = a * b, field multiplication

    b = b_int % p_int
    """
    b = b_int % CHARACTERISTIC
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0 or b == 0:
        return 0

    return EXP[m + n]


def _power_lookup(a, b_int):  # pragma: no cover
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
    m = LOG[a]

    if b_int == 0:
        return 1

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return 0

    return EXP[(m * b_int) % (ORDER - 1)]


def _log_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    log(a, α) = log(α^m, α)
              = m
    """
    return LOG[a]


def _poly_eval_lookup(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
