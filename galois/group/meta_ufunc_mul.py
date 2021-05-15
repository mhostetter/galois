import numba
import numpy as np

from ..meta_ufunc import Ufunc

MODULUS = None  # The modulus of the group, not the order
ORDER = None  # The size of the group

RECIPROCAL_UFUNC = lambda x: 1 / x


class MultiplicativeGroupUfunc(Ufunc):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    _overridden_ufuncs = {
        np.multiply: "_ufunc_multiply",
        np.reciprocal: "_ufunc_reciprocal",
        np.power: "_ufunc_power",
        np.square: "_ufunc_square",
    }

    _unsupported_ufuncs = Ufunc._unsupported_ufuncs + [
        np.add,
        np.negative,
        np.subtract,
        np.floor_divide,
        np.true_divide,
        np.log,
        np.matmul,
    ]

    def _compile_ufuncs(cls, target):
        global MODULUS, ORDER, RECIPROCAL_UFUNC

        if cls._ufunc_mode == "jit-calculate":
            MODULUS = cls.modulus
            ORDER = cls.order
            kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
            cls._ufuncs["multiply"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
            cls._ufuncs["reciprocal"] = numba.vectorize(["int64(int64)"], **kwargs)(_reciprocal_calculate)
            RECIPROCAL_UFUNC = cls._ufuncs["reciprocal"]
            cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)

        else:
            cls._ufuncs["multiply"] = np.frompyfunc(cls._multiply_python, 2, 1)
            cls._ufuncs["reciprocal"] = np.frompyfunc(cls._reciprocal_python, 1, 1)
            cls._ufuncs["power"] = np.frompyfunc(cls._power_python, 2, 1)

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    def _ufunc_multiply(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["multiply"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        output = getattr(cls._ufuncs["reciprocal"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_power(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_binary_method_not_reduction(ufunc, method)
        cls._verify_operands_first_field_second_int(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["power"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        inputs = list(inputs) + [2]
        return cls._ufunc_power(ufunc, method, inputs, kwargs, meta)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _multiply_python(cls, a, b):
        return (a * b) % cls.modulus

    def _reciprocal_python(cls, a):
        """
        s*x + t*y = gcd(x, y) = 1
        x = p
        y = a in GF(p)
        t = a**-1 in GF(p)
        """
        r2, r1 = cls.modulus, a
        t2, t1 = 0, 1

        while r1 != 0:
            q = r2 // r1
            r2, r1 = r1, r2 - q*r1
            t2, t1 = t1, t2 - q*t1

        if t2 < 0:
            t2 += cls.modulus

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
        if power < 0:
            a = cls._reciprocal_python(a)
            power = abs(power)

        return pow(a, power, cls.modulus)


###############################################################################
# Group arithmetic, explicitly calculated without lookup tables
###############################################################################

def _multiply_calculate(a, b):  # pragma: no cover
    return (a * b) % MODULUS


def _reciprocal_calculate(a):  # pragma: no cover
    r2, r1 = MODULUS, a
    t2, t1 = 0, 1

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q*r1
        t2, t1 = t1, t2 - q*t1

    if t2 < 0:
        t2 += MODULUS

    return t2


def _power_calculate(a, power):  # pragma: no cover
    if power == 0:
        return 1
    elif power < 0:
        a = RECIPROCAL_UFUNC(a)
        power = abs(power)

    if power > ORDER:
        power = power % (ORDER)

    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = (result_s * result_s) % MODULUS
            power //= 2
        else:
            result_m = (result_m * result_s) % MODULUS
            power -= 1

    result = (result_m * result_s) % MODULUS

    return result
