"""
A module that contains a metaclass mixin that provides NumPy ufunc overriding for an ndarray subclass.
"""
import numpy as np

from ._calculate import CalculateMeta
from ._lookup import LookupMeta


class UfuncMeta(LookupMeta, CalculateMeta):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter,abstract-method

    _UNSUPPORTED_UFUNCS_UNARY = [
        np.invert, np.sqrt,
        np.log2, np.log10,
        np.exp, np.expm1, np.exp2,
        np.sin, np.cos, np.tan,
        np.sinh, np.cosh, np.tanh,
        np.arcsin, np.arccos, np.arctan,
        np.arcsinh, np.arccosh, np.arctanh,
        np.degrees, np.radians,
        np.deg2rad, np.rad2deg,
        np.floor, np.ceil, np.trunc, np.rint,
    ]

    _UNSUPPORTED_UFUNCS_BINARY = [
        np.hypot, np.arctan2,
        np.logaddexp, np.logaddexp2,
        np.fmod, np.modf,
        np.fmin, np.fmax,
    ]

    _UNSUPPORTED_UFUNCS = _UNSUPPORTED_UFUNCS_UNARY + _UNSUPPORTED_UFUNCS_BINARY

    _UFUNCS_REQUIRING_VIEW = [
        np.bitwise_and, np.bitwise_or, np.bitwise_xor,
        np.left_shift, np.right_shift,
        np.positive,
    ]

    _OVERRIDDEN_UFUNCS = {
        np.add: "_ufunc_routine_add",
        np.negative: "_ufunc_routine_negative",
        np.subtract: "_ufunc_routine_subtract",
        np.multiply: "_ufunc_routine_multiply",
        np.reciprocal: "_ufunc_routine_reciprocal",
        np.floor_divide: "_ufunc_routine_divide",
        np.true_divide: "_ufunc_routine_divide",
        np.divmod: "_ufunc_routine_divmod",
        np.remainder: "_ufunc_routine_remainder",
        np.power: "_ufunc_routine_power",
        np.square: "_ufunc_routine_square",
        np.log: "_ufunc_routine_log",
        np.matmul: "_ufunc_routine_matmul",
    }

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._ufuncs = {}

    def _compile_ufuncs(cls):
        """
        Compile/re-compile the ufuncs based on the `ufunc_mode`. This may be supplemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        cls._ufuncs = {}  # Reset the dictionary so each ufunc will get recompiled

        if cls.ufunc_mode == "jit-lookup":
            cls._build_lookup_tables()

    def _ufunc(cls, name):
        """
        Returns the ufunc for the specific type of arithmetic. The ufunc compilation is based on `ufunc_mode`.
        """
        if name not in cls._ufuncs:
            if cls.ufunc_mode == "jit-lookup":
                cls._ufuncs[name] = cls._ufunc_lookup(name)
            elif cls.ufunc_mode == "jit-calculate":
                cls._ufuncs[name] = cls._ufunc_calculate(name)
            else:
                cls._ufuncs[name] = cls._ufunc_python(name)
        return cls._ufuncs[name]

    ###############################################################################
    # Input/output conversion functions
    ###############################################################################

    def _verify_unary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat", "outer"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Reduction methods are only supported on binary functions.")

    def _verify_binary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r} because it takes inputs with type {cls.name} array and integer array. Different types do not support reduction.")

    def _verify_method_only_call(cls, ufunc, method):  # pylint: disable=no-self-use
        if not method == "__call__":
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Only '__call__' is supported.")

    def _verify_operands_in_same_field(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["non_field_operands"]) > 0:
            raise TypeError(f"Operation {ufunc.__name__!r} requires both operands to be {cls.name} arrays, not {[inputs[i] for i in meta['operands']]}.")

    def _verify_operands_in_field_or_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        for i in meta["non_field_operands"]:
            if isinstance(inputs[i], (int, np.integer)):
                pass
            elif isinstance(inputs[i], np.ndarray):
                if meta["field"].dtypes == [np.object_]:
                    if not (inputs[i].dtype == np.object_ or np.issubdtype(inputs[i].dtype, np.integer)):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
                else:
                    if not np.issubdtype(inputs[i].dtype, np.integer):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
            else:
                raise TypeError(f"Operation {ufunc.__name__!r} requires operands that are not {cls.name} arrays to be integers or an integer np.ndarray, not {type(inputs[i])}.")

    def _verify_operands_first_field_second_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["operands"]) == 1:
            return

        if not meta["operands"][0] == meta["field_operands"][0]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the first operand to be a {cls.name} array, not {meta['types'][meta['operands'][0]]}.")
        if len(meta["field_operands"]) > 1 and meta["operands"][1] == meta["field_operands"][1]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer array, not {meta['types'][meta['operands'][1]]}.")

        second = inputs[meta["operands"][1]]
        if isinstance(second, (int, np.integer)):
            return
        # elif type(second) is np.ndarray:
        #     if not np.issubdtype(second.dtype, np.integer):
        #         raise ValueError(f"Operation {ufunc.__name__!r} requires the second operand with type np.ndarray to have integer dtype, not {second.dtype}.")
        elif isinstance(second, np.ndarray):
            if meta["field"].dtypes == [np.object_]:
                if not (second.dtype == np.object_ or np.issubdtype(second.dtype, np.integer)):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
            else:
                if not np.issubdtype(second.dtype, np.integer):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
        else:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer or integer np.ndarray, not {type(second)}.")

    def _view_inputs_as_ndarray(cls, inputs, kwargs, dtype=None):  # pylint: disable=no-self-use
        # View all inputs that are Galois field arrays as np.ndarray to avoid infinite recursion
        v_inputs = list(inputs)
        for i in range(len(inputs)):
            if issubclass(type(inputs[i]), cls):
                v_inputs[i] = inputs[i].view(np.ndarray) if dtype is None else inputs[i].view(np.ndarray).astype(dtype)

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if issubclass(type(output), cls):
                    o = output.view(np.ndarray) if dtype is None else output.view(np.ndarray).astype(dtype)
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    def _view_output_as_field(cls, output, field, dtype):  # pylint: disable=no-self-use
        if isinstance(type(output), field):
            return output
        elif isinstance(output, np.ndarray):
            return output.astype(dtype).view(field)
        elif output is None:
            return None
        else:
            return field(output, dtype=dtype)

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    def _ufunc_routine_add(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("add"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("negative"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_subtract(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("subtract"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_multiply(cls, ufunc, method, inputs, kwargs, meta):
        if len(meta["non_field_operands"]) > 0:
            # Scalar multiplication
            cls._verify_operands_in_field_or_int(ufunc, inputs, meta)
            inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
            inputs[meta["non_field_operands"][0]] = np.mod(inputs[meta["non_field_operands"][0]], cls.characteristic)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("multiply"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("reciprocal"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_divide(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("divide"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_divmod(cls, ufunc, method, inputs, kwargs, meta):
        q = cls._ufunc_routine_divide(ufunc, method, inputs, kwargs, meta)
        r = cls.Zeros(q.shape)
        output = q, r
        return output

    def _ufunc_routine_remainder(cls, ufunc, method, inputs, kwargs, meta):
        # Perform dummy addition operation to get shape of output zeros
        x = cls._ufunc_routine_add(ufunc, method, inputs, kwargs, meta)
        output = cls.Zeros(x.shape)
        return output

    def _ufunc_routine_power(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_binary_method_not_reduction(ufunc, method)
        cls._verify_operands_first_field_second_int(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("power"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs = list(inputs) + [2]
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("power"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        inputs = list(inputs) + [int(cls.primitive_element)]
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("log"), method)(*inputs, **kwargs)
        return output

    def _ufunc_routine_matmul(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        return cls._matmul(*inputs, **kwargs)
