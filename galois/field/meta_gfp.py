import numba
import numpy as np

from ..dtypes import DTYPES
from ..modular import primitive_root

from .meta import FieldMeta
from .poly import Poly

CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
ORDER = None  # The field's order `p^m`
PRIMITIVE_ELEMENT = None  # The field's primitive element

RECIPROCAL_UFUNC = lambda x: 1 / x


class GFpMeta(FieldMeta):
    """
    An abstract base class for all :math:`\\mathrm{GF}(p)` field array classes.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._prime_subfield = cls
        cls._is_primitive_poly = True

        cls.compile(kwargs["mode"], kwargs["target"])

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

    def _compile_ufuncs(cls, target):
        global CHARACTERISTIC, ORDER, PRIMITIVE_ELEMENT, RECIPROCAL_UFUNC

        if cls._ufunc_mode == "jit-lookup":
            cls._build_lookup_tables()

            # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835548405.
            ORDER = cls.order
            kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
            cls._ufuncs["add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_calculate)
            cls._ufuncs["negative"] = numba.vectorize(["int64(int64)"], **kwargs)(_negative_calculate)
            cls._ufuncs["subtract"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_calculate)

            cls._ufuncs["multiply"] = cls._compile_multiply_lookup(target)
            cls._ufuncs["reciprocal"] = cls._compile_reciprocal_lookup(target)
            cls._ufuncs["divide"] = cls._compile_divide_lookup(target)
            cls._ufuncs["power"] = cls._compile_power_lookup(target)
            cls._ufuncs["log"] = cls._compile_log_lookup(target)

        elif cls._ufunc_mode == "jit-calculate":
            CHARACTERISTIC = cls.characteristic
            ORDER = cls.order
            PRIMITIVE_ELEMENT = int(cls.primitive_element)  # Convert from field element to integer

            kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
            cls._ufuncs["add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_calculate)
            cls._ufuncs["negative"] = numba.vectorize(["int64(int64)"], **kwargs)(_negative_calculate)
            cls._ufuncs["subtract"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_calculate)
            cls._ufuncs["multiply"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
            cls._ufuncs["reciprocal"] = numba.vectorize(["int64(int64)"], **kwargs)(_reciprocal_calculate)
            RECIPROCAL_UFUNC = cls._ufuncs["reciprocal"]
            cls._ufuncs["divide"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_calculate)
            cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
            cls._ufuncs["log"] = numba.vectorize(["int64(int64)"], **kwargs)(_log_calculate)

        else:
            cls._ufuncs["add"] = np.frompyfunc(cls._add_python, 2, 1)
            cls._ufuncs["negative"] = np.frompyfunc(cls._negative_python, 1, 1)
            cls._ufuncs["subtract"] = np.frompyfunc(cls._subtract_python, 2, 1)
            cls._ufuncs["multiply"] = np.frompyfunc(cls._multiply_python, 2, 1)
            cls._ufuncs["reciprocal"] = np.frompyfunc(cls._reciprocal_python, 1, 1)
            cls._ufuncs["divide"] = np.frompyfunc(cls._divide_python, 2, 1)
            cls._ufuncs["power"] = np.frompyfunc(cls._power_python, 2, 1)
            cls._ufuncs["log"] = np.frompyfunc(cls._log_python, 1, 1)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
        c = a + b
        if c >= cls.order:  # pylint: disable=comparison-with-callable
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
        """
        s*x + t*y = gcd(x, y) = 1
        x = p
        y = a in GF(p)
        t = a**-1 in GF(p)
        """
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
        if a == 0 and power < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if power == 0:
            return 1
        elif power < 0:
            a = cls._reciprocal_python(a)
            power = abs(power)

        # In GF(p), we can reduce the power mod p-1 since a^(p-1) = 1 (mod p)
        if power > cls.order - 1:
            power = power % (cls.order - 1)

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

def _add_calculate(a, b):  # pragma: no cover
    c = a + b
    if c >= ORDER:
        c -= ORDER
    return c


def _negative_calculate(a):  # pragma: no cover
    if a == 0:
        return 0
    else:
        return ORDER - a


def _subtract_calculate(a, b):  # pragma: no cover
    if a >= b:
        return a - b
    else:
        return ORDER + a - b


def _multiply_calculate(a, b):  # pragma: no cover
    return (a * b) % ORDER


def _reciprocal_calculate(a):  # pragma: no cover
    """
    s*x + t*y = gcd(x, y) = 1
    x = p
    y = a in GF(p)
    t = a**-1 in GF(p)
    """
    if a == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    r2, r1 = ORDER, a
    t2, t1 = 0, 1

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q*r1
        t2, t1 = t1, t2 - q*t1

    if t2 < 0:
        t2 += ORDER

    return t2


def _divide_calculate(a, b):  # pragma: no cover
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:
        return 0
    else:
        b_inv = RECIPROCAL_UFUNC(b)
        return (a * b_inv) % ORDER


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

    # In GF(p), we can reduce the power mod p-1 since a^(p-1) = 1 (mod p)
    if power > ORDER - 1:
        power = power % (ORDER - 1)

    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = (result_s * result_s) % ORDER
            power //= 2
        else:
            result_m = (result_m * result_s) % ORDER
            power -= 1

    result = (result_m * result_s) % ORDER

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
        result = (result * PRIMITIVE_ELEMENT) % ORDER

    return i
