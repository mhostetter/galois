import numba
import numpy as np

from ..meta_ufunc import Ufunc

MODULUS = None  # The modulus of the group, not the order
ORDER = None  # The size of the group


class AdditiveGroupUfunc(Ufunc):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    _overridden_ufuncs = {
        np.add: "_ufunc_routine_add",
        np.negative: "_ufunc_routine_negative",
        np.power: "_ufunc_routine_power",
        np.square: "_ufunc_routine_square",
        # np.log: "_ufunc_routine_log",
    }

    _unsupported_ufuncs = Ufunc._unsupported_ufuncs + [
        np.subtract,
        np.multiply,
        np.reciprocal,
        np.floor_divide,
        np.true_divide,
        np.log,
        np.matmul,
    ]

    ###############################################################################
    # Compile general-purpose calculate functions
    ###############################################################################

    def _compile_power_calculate(cls):
        global MODULUS, ORDER
        MODULUS = cls.modulus
        ORDER = cls.order
        kwargs = {"nopython": True, "target": cls.ufunc_target} if cls.ufunc_target != "cuda" else {"target": cls.ufunc_target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)

    ###############################################################################
    # Individual ufuncs, compiled on-demand
    ###############################################################################

    def _ufunc_power(cls):
        if cls._ufuncs.get("power", None) is None:
            if cls.ufunc_mode == "jit-calculate":
                cls._ufuncs["power"] = cls._compile_power_calculate()
            else:
                cls._ufuncs["power"] = np.frompyfunc(cls._power_python, 2, 1)
        return cls._ufuncs["power"]

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    def _ufunc_routine_add(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(np.add, method)(*inputs, **kwargs)
        output = np.mod(output, cls.modulus)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = np.mod(cls.modulus - inputs[0], cls.modulus)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_power(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_binary_method_not_reduction(ufunc, method)
        cls._verify_operands_first_field_second_int(ufunc, inputs, meta)
        output = getattr(cls._ufunc_power(), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        inputs = list(inputs) + [2]
        return cls._ufunc_power()(ufunc, method, inputs, kwargs, meta)

    # def _ufunc_routine_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
    #     cls._verify_method_only_call(ufunc, method)
    #     # base = cls.generator if len(inputs) == 1 else inputs[1]
    #     # base = np.broadcast_to(base, inputs[0].shape)
    #     output = getattr(cls._ufunc_log(), method)(*inputs, **kwargs)
    #     return output

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _power_python(cls, a, power):
        return (a * power) % cls.modulus

    # def _log_python(cls, alpha, beta):
    #     r2, r1 = cls.order, beta
    #     t2, t1 = 0, 1

    #     while r1 != 0:
    #         q = r2 // r1
    #         r2, r1 = r1, r2 - q*r1
    #         t2, t1 = t1, t2 - q*t1

    #     # t2 = 1 (mod ORDER)
    #     t2 = t2 * alpha

    #     return t2 % cls.order


###############################################################################
# Additive group arithmetic, explicitly calculated without lookup tables
###############################################################################

def _power_calculate(a, power):
    if power > MODULUS:
        power = power % MODULUS
    return (a * power) % MODULUS


# def _log_calculate(alpha, beta):
#     r2, r1 = ORDER, beta
#     t2, t1 = 0, 1

#     while r1 != 0:
#         q = r2 // r1
#         r2, r1 = r1, r2 - q*r1
#         t2, t1 = t1, t2 - q*t1

#     # t2 = 1 (mod ORDER)
#     t2 = t2 * alpha

#     return t2 % ORDER
