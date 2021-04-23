import numpy as np

from .linalg import matmul
from .meta_gf import GFMeta

###############################################################################
# Overridden ufunc routines
###############################################################################

# NOTE: These are functions, not methods, because we don't want all these private
# methods being carried around by the np.ndarray subclass.

def _ufunc_add(ufunc, method, inputs, kwargs, meta):
    _verify_operands_in_same_field(ufunc, inputs, meta)
    output = getattr(meta["ufuncs"]["add"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_subtract(ufunc, method, inputs, kwargs, meta):
    _verify_operands_in_same_field(ufunc, inputs, meta)
    output = getattr(meta["ufuncs"]["subtract"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_multiply(ufunc, method, inputs, kwargs, meta):
    if len(meta["non_field_operands"]) == 0:
        # In-field multiplication
        output = getattr(meta["ufuncs"]["multiply"], method)(*inputs, **kwargs)
    else:
        # Scalar multiplication
        inputs = _verify_and_flip_operands_first_field_second_int(ufunc, method, inputs, meta)
        output = getattr(meta["ufuncs"]["multiple_add"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_divide(ufunc, method, inputs, kwargs, meta):
    _verify_operands_in_same_field(ufunc, inputs, meta)
    if np.count_nonzero(inputs[meta["operands"][-1]]) != inputs[meta["operands"][-1]].size:
        raise ZeroDivisionError("Cannot divide by 0 in Galois fields.")
    output = getattr(meta["ufuncs"]["divide"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_negative(ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
    output = getattr(meta["ufuncs"]["negative"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_reciprocal(ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
    output = getattr(meta["ufuncs"]["reciprocal"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_power(ufunc, method, inputs, kwargs, meta):
    _verify_operands_first_field_second_int(ufunc, inputs, meta)
    if method == "outer" and (np.any(inputs[meta["operands"][0]] == 0) and np.any(inputs[meta["operands"][1]] < 0)):
        raise ZeroDivisionError("Cannot take the multiplicative inverse of 0 in Galois fields.")
    if method == "__call__" and np.any(np.logical_and(inputs[meta["operands"][0]] == 0, inputs[meta["operands"][1]] < 0)):
        raise ZeroDivisionError("Cannot take the multiplicative inverse of 0 in Galois fields.")
    output = getattr(meta["ufuncs"]["power"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_square(ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
    inputs = list(inputs)
    inputs.append(2)
    output = getattr(meta["ufuncs"]["power"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


def _ufunc_log(ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
    if np.count_nonzero(inputs[meta["operands"][0]]) != inputs[meta["operands"][0]].size:
        raise ArithmeticError("Cannot take the logarithm of 0 in Galois fields.")
    output = getattr(meta["ufuncs"]["log"], method)(*inputs, **kwargs)
    output = _view_output_as_field(output, meta["field"], meta["dtype"])
    return output


###############################################################################
# Input conversion and type verification functions
###############################################################################

def _verify_operands_in_same_field(ufunc, inputs, meta):
    if len(meta["non_field_operands"]) > 0:
        raise TypeError(f"Operation '{ufunc.__name__}' requires both operands to be Galois field arrays over the same field, not {[inputs[i] for i in meta['operands']]}.")


def _verify_and_flip_operands_first_field_second_int(ufunc, method, inputs, meta):
    if len(meta["operands"]) == 1 or method == "reduceat":
        return inputs

    assert len(meta["non_field_operands"]) == 1
    i = meta["non_field_operands"][0]

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
        raise TypeError(f"Operation '{ufunc.__name__}' requires operands that are not Galois field arrays to be an integers or integer np.ndarrays, not {type(inputs[i])}.")

    if meta["operands"][0] == meta["field_operands"][0]:
        # If the Galois field array is the first argument, continue
        pass
    else:
        # Switch arguments to guarantee the Galois field array is the first element
        inputs = list(inputs)
        inputs[meta["operands"][0]], inputs[meta["operands"][1]] = inputs[meta["operands"][1]], inputs[meta["operands"][0]]

    return inputs


def _verify_operands_first_field_second_int(ufunc, inputs, meta):
    if len(meta["operands"]) == 1:
        return
    first = inputs[meta["operands"][0]]
    second = inputs[meta["operands"][1]]

    if not meta["operands"][0] == meta["field_operands"][0]:
        raise TypeError(f"Operation '{ufunc.__name__}' requires the first operand to be a Galois field array, not {first}.")

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


def _view_inputs_as_ndarray(inputs, kwargs):
    # View all inputs that are Galois field arrays as np.ndarray to avoid infinite recursion
    v_inputs = list(inputs)
    for i in range(len(inputs)):
        if isinstance(type(inputs[i]), GFMeta):
            v_inputs[i] = inputs[i].view(np.ndarray)

    # View all output arrays as np.ndarray to avoid infinite recursion
    if "out" in kwargs:
        outputs = kwargs["out"]
        v_outputs = []
        for output in outputs:
            if isinstance(type(output), GFMeta):
                o = output.view(np.ndarray)
            else:
                o = output
            v_outputs.append(o)
        kwargs["out"] = tuple(v_outputs)

    return v_inputs, kwargs


def _view_output_as_field(output, field, dtype):
    if isinstance(output, np.ndarray):
        return output.astype(dtype).view(field)
    elif output is None:
        return None
    else:
        return field(output)


###############################################################################
# GFArray mixin class
###############################################################################

OVERRIDDEN_UFUNCS = {
    np.add: _ufunc_add,
    np.subtract: _ufunc_subtract,
    np.multiply: _ufunc_multiply,
    np.floor_divide: _ufunc_divide,
    np.true_divide: _ufunc_divide,
    np.negative: _ufunc_negative,
    np.reciprocal: _ufunc_reciprocal,
    np.power: _ufunc_power,
    np.square: _ufunc_square,
    np.log: _ufunc_log,
}

UNSUPPORTED_ONE_ARG_UFUNCS = [
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

UNSUPPORTED_TWO_ARG_UFUNCS = [
    np.hypot, np.arctan2,
    np.logaddexp, np.logaddexp2,
    np.remainder,
]

UNSUPPORTED_UFUNCS = UNSUPPORTED_ONE_ARG_UFUNCS + UNSUPPORTED_TWO_ARG_UFUNCS


class UfuncMixin(np.ndarray):
    """
    A mixin class that provides the overridding numpy ufunc functionality.
    """

    # def _view_output_ndarray_as_field(self, ufunc, v_outputs):
    #     if v_outputs is NotImplemented:
    #         return v_outputs
    #     if ufunc.nout == 1:
    #         v_outputs = (v_outputs, )

    #     outputs = []
    #     for v_output in v_outputs:
    #         o = self.__class__(v_output, dtype=self.dtype)
    #         outputs.append(o)

    #     return outputs[0] if len(outputs) == 1 else outputs

    # def _verify_inputs(self, ufunc, method, inputs, meta):  # pylint: disable=too-many-branches
    #     types = [meta["types"][i] for i in meta["operands"]]  # List of types of the "operands", excludes index lists, etc
    #     operands = [inputs[i] for i in meta["operands"]]

    #     if method == "reduceat":
    #         return

    #     # Verify input operand types
    #     if ufunc in [np.add, np.subtract, np.true_divide, np.floor_divide]:
    #         if not all(t is self.__class__ for t in types):
    #             raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the same field {self.__class__}, not {types}")
    #     if ufunc in [np.multiply, np.power, np.square]:
    #         if not all(np.issubdtype(o.dtype, np.integer) or o.dtype == np.object_ for o in operands):
    #             raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the field {self.__class__} or integers, not {types}")
    #     if ufunc in [np.power, np.square]:
    #         if not types[0] is self.__class__:
    #             raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields can only exponentiate elements in the same field {self.__class__}, not {types[0]}")

    #     # Verify no divide by zero or log(0) errors
    #     if ufunc in [np.true_divide, np.floor_divide] and np.count_nonzero(operands[-1]) != operands[-1].size:
    #         raise ZeroDivisionError("Divide by 0")
    #     if ufunc is np.power:
    #         if method == "outer" and (np.any(operands[0] == 0) and np.any(operands[1] < 0)):
    #             raise ZeroDivisionError("Divide by 0")
    #         if method == "__call__" and np.any(np.logical_and(operands[0] == 0, operands[1] < 0)):
    #             raise ZeroDivisionError("Divide by 0")
    #     if ufunc is np.log and np.count_nonzero(operands[0]) != operands[0].size:
    #         raise ArithmeticError("Log(0) error")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # pylint: disable=too-many-branches
        """
        Intercept various numpy ufuncs (triggered by operators like `+` , `-`, etc). Then determine
        which operations will result in the correct answer in the given Galois field. Wherever
        appropriate, use native numpy ufuncs for their efficiency and generality in supporting various array
        shapes, etc.
        """
        if ufunc is np.matmul:
            return matmul(*inputs, **kwargs)

        meta = {}
        meta["types"] = [type(inputs[i]) for i in range(len(inputs))]
        meta["operands"] = list(range(len(inputs)))
        if method in ["at", "reduceat"]:
            # Remove the second argument for "at" ufuncs which is the indices list
            meta["operands"].pop(1)
        meta["field_operands"] = [i for i in meta["operands"] if isinstance(inputs[i], self.__class__)]
        meta["non_field_operands"] = [i for i in meta["operands"] if not isinstance(inputs[i], self.__class__)]
        meta["field"] = self.__class__
        meta["dtype"] = self.dtype
        meta["ufuncs"] = self._ufuncs

        inputs, kwargs = _view_inputs_as_ndarray(inputs, kwargs)

        if ufunc in OVERRIDDEN_UFUNCS:
            # Set all ufuncs with "casting" keyword argument to "unsafe" so we can cast unsigned integers
            # to integers. We know this is safe because we already verified the inputs.
            if method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            # Need to set the intermediate dtype for reduction operations or an error will be thrown. We
            # use the largest valid dtype for this field.
            if method in ["reduce"]:
                kwargs["dtype"] = type(self).dtypes[-1]

            return OVERRIDDEN_UFUNCS[ufunc](ufunc, method, inputs, kwargs, meta)

        elif ufunc in UNSUPPORTED_UFUNCS:
            raise NotImplementedError(f"The numpy ufunc '{ufunc.__name__}' is not supported on Galois field arrays. If you believe this ufunc should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        else:
            return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member
