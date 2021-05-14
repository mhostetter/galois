import numpy as np

from ..meta_ufunc import Ufunc


class AdditiveGroupUfunc(Ufunc):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    _overridden_ufuncs = {
        np.add: "_ufunc_add",
        np.negative: "_ufunc_negative",
    }

    _unsupported_ufuncs = Ufunc._unsupported_ufuncs + [
        np.subtract,
        np.multiply,
        np.reciprocal,
        np.floor_divide,
        np.true_divide,
        np.power,
        np.square,
        np.log,
        np.matmul,
    ]

    def _compile_ufuncs(cls, target):
        return

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    def _ufunc_add(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(np.add, method)(*inputs, **kwargs)
        output = np.mod(output, cls.modulus)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = np.mod(cls.modulus - inputs[0], cls.modulus)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output
