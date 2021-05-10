import numpy as np

# List of ufuncs that are not valid on arrays over finite groups, rings, and fields
UNSUPPORTED_UFUNCS_UNARY = [
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

UNSUPPORTED_UFUNCS_BINARY = [
    np.hypot, np.arctan2,
    np.logaddexp, np.logaddexp2,
    np.remainder,
    np.fmod, np.modf,
    np.fmin, np.fmax,
]

UFUNCS_REQUIRING_VIEW = [
    np.bitwise_and, np.bitwise_or, np.bitwise_xor,
    np.left_shift, np.right_shift,
    np.positive,
]


class Ufunc(type):
    """
    A base class for :obj:`GroupUfunc`, :obj:`RingUfunc`, and :obj:`FieldUfunc`.
    """
    # pylint: disable=no-value-for-parameter

    _unsupported_ufuncs = UNSUPPORTED_UFUNCS_UNARY + UNSUPPORTED_UFUNCS_BINARY
    _ufuncs_requiring_view = UFUNCS_REQUIRING_VIEW
    _overridden_ufuncs = {}

    def _compile_ufuncs(cls, target):
        raise NotImplementedError

    ###############################################################################
    # Input/output conversion functions
    ###############################################################################

    def _verify_unary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat", "outer"]:
            raise ValueError(f"Ufunc method '{method}' is not supported on '{ufunc.__name__}'. Reduction methods are only supported on binary functions.")

    def _verify_binary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat"]:
            raise ValueError(f"Ufunc method '{method}' is not supported on '{ufunc.__name__}' because it takes inputs with type {cls.name} array and integer array. Different types do not support reduction.")

    def _verify_method_only_call(cls, ufunc, method):  # pylint: disable=no-self-use
        if not method == "__call__":
            raise ValueError(f"Ufunc method '{method}' is not supported on '{ufunc.__name__}'. Only '__call__' is supported.")

    def _verify_operands_in_same_field(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["non_field_operands"]) > 0:
            raise TypeError(f"Operation '{ufunc.__name__}' requires both operands to be {cls.name} arrays, not {[inputs[i] for i in meta['operands']]}.")

    def _verify_operands_in_field_or_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        for i in meta["non_field_operands"]:
            if isinstance(inputs[i], (int, np.integer)):
                pass
            elif isinstance(inputs[i], np.ndarray):
                if meta["field"].dtypes == [np.object_]:
                    if not (inputs[i].dtype == np.object_ or np.issubdtype(inputs[i].dtype, np.integer)):
                        raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{inputs[i].dtype}'.")
                else:
                    if not np.issubdtype(inputs[i].dtype, np.integer):
                        raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{inputs[i].dtype}'.")
            else:
                raise TypeError(f"Operation '{ufunc.__name__}' requires operands that are not {cls.name} arrays to be integers or an integer np.ndarray, not {type(inputs[i])}.")

    def _verify_operands_first_field_second_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["operands"]) == 1:
            return

        if not meta["operands"][0] == meta["field_operands"][0]:
            raise TypeError(f"Operation '{ufunc.__name__}' requires the first operand to be a {cls.name} array, not {meta['types'][meta['operands'][0]]}.")
        if len(meta["field_operands"]) > 1 and meta["operands"][1] == meta["field_operands"][1]:
            raise TypeError(f"Operation '{ufunc.__name__}' requires the second operand to be an integer array, not {meta['types'][meta['operands'][1]]}.")

        second = inputs[meta["operands"][1]]
        if isinstance(second, (int, np.integer)):
            return
        # elif type(second) is np.ndarray:
        #     if not np.issubdtype(second.dtype, np.integer):
        #         raise ValueError(f"Operation '{ufunc.__name__}' requires the second operand with type np.ndarray to have integer dtype, not '{second.dtype}'.")
        elif isinstance(second, np.ndarray):
            if meta["field"].dtypes == [np.object_]:
                if not (second.dtype == np.object_ or np.issubdtype(second.dtype, np.integer)):
                    raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{second.dtype}'.")
            else:
                if not np.issubdtype(second.dtype, np.integer):
                    raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{second.dtype}'.")
        else:
            raise TypeError(f"Operation '{ufunc.__name__}' requires the second operand to be an integer or integer np.ndarray, not {type(second)}.")

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
