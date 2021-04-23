import numba
import numpy as np

from .algorithm import gcd, gcd_jit
from .dtypes import DTYPES
from .meta_gf import GFMeta
from .meta_mixin_prime_field import PrimeFieldMixin
from .poly import Poly

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
ORDER = None  # The field's order `p^m`
ALPHA = None  # The field's primitive element

# Placeholder functions to be replaced by JIT-compiled function
ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y
MULTIPLICATIVE_INVERSE_JIT = lambda x: 1 / x


class GFpMeta(GFMeta, PrimeFieldMixin):
    """
    An abstract base class for all :math:`\\mathrm{GF}(p)` field array classes.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._characteristic = kwargs["characteristic"]
        cls._degree = 1
        cls._order = cls.characteristic**cls.degree
        cls._primitive_element = kwargs["primitive_element"]
        cls._ground_field = cls

        cls.compile(kwargs["mode"], kwargs["target"])

        cls._irreducible_poly = Poly([1, -int(cls.primitive_element)], field=cls)
        cls._primitive_element = cls(cls.primitive_element)
        cls._is_primitive_poly = True

        # Add helper variables for python ufuncs. This prevents the ufuncs from having to repeatedly calculate them.
        cls._primitive_element_dec = int(cls.primitive_element)
        cls._irreducible_poly_dec = cls.irreducible_poly.integer  # pylint: disable=no-member

    @property
    def dtypes(cls):
        max_dtype = DTYPES[-1]
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(d) == 0:
            d = [np.object_]
        return d

    def _compile_jit_calculate(cls, target):
        global CHARACTERISTIC, ORDER, ALPHA, ADD_JIT, MULTIPLY_JIT, MULTIPLICATIVE_INVERSE_JIT
        CHARACTERISTIC = cls.characteristic
        ORDER = cls.order
        ALPHA = int(cls.primitive_element)  # Convert from field element to integer

        # JIT-compile add, multiply, and multiplicative inverse routines for reference in polynomial evaluation routine
        ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(_add_calculate)
        MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(_multiply_calculate)
        MULTIPLICATIVE_INVERSE_JIT = numba.jit("int64(int64)", nopython=True)(_multiplicative_inverse_calculate)

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # Create numba JIT-compiled ufuncs
        cls._ufuncs["add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_calculate)
        cls._ufuncs["subtract"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_calculate)
        cls._ufuncs["multiply"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
        cls._ufuncs["divide"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_calculate)
        cls._ufuncs["negative"] = numba.vectorize(["int64(int64)"], **kwargs)(_additive_inverse_calculate)
        cls._ufuncs["reciprocal"] = numba.vectorize(["int64(int64)"], **kwargs)(_multiplicative_inverse_calculate)
        cls._ufuncs["multiple_add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiple_add_calculate)
        cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
        cls._ufuncs["log"] = numba.vectorize(["int64(int64)"], **kwargs)(_log_calculate)
        cls._ufuncs["poly_eval"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_calculate)

    ###############################################################################
    # Overridden methods from ObjectCalculateMixin
    ###############################################################################

    def _add_python(cls, a, b):
        return (a + b) % cls.order

    def _subtract_python(cls, a, b):
        return (a - b) % cls.order

    def _multiply_python(cls, a, b):
        return (a * b) % cls.order

    def _additive_inverse_python(cls, a):
        return (-a) % cls.order

    def _multiplicative_inverse_python(cls, a):
        a_inv = gcd(a, cls.order)[1]
        return a_inv % cls.order


###############################################################################
# Galois field arithmetic, explicitly calculated without lookup tables
###############################################################################

def _add_calculate(a, b):  # pragma: no cover
    return (a + b) % ORDER


def _subtract_calculate(a, b):  # pragma: no cover
    return (a - b) % ORDER


def _multiply_calculate(a, b):  # pragma: no cover
    return (a * b) % ORDER


def _divide_calculate(a, b):  # pragma: no cover
    if a == 0 or b == 0:
        # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0
    b_inv = MULTIPLICATIVE_INVERSE_JIT(b)
    return MULTIPLY_JIT(a, b_inv)


def _additive_inverse_calculate(a):  # pragma: no cover
    return (-a) % ORDER


def _multiplicative_inverse_calculate(a):  # pragma: no cover
    a_inv = gcd_jit(a, ORDER)[1]
    return a_inv % ORDER


def _multiple_add_calculate(a, multiple):  # pragma: no cover
    return (a * multiple) % ORDER


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
    # NOTE: The a == 0 and b < 0 condition will be caught outside of the the ufunc and raise ZeroDivisonError
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
    # Naive algorithm
    result = 1
    for i in range(0, ORDER-1):
        if result == beta:
            break
        result = MULTIPLY_JIT(result, ALPHA)
    return i


def _poly_eval_calculate(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
