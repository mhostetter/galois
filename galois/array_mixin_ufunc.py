import numpy as np

# Dictionary mapping numpy ufuncs to our implementation method
OVERRIDDEN_UFUNCS = [
    np.add,
    np.subtract,
    np.multiply,
    np.floor_divide,
    np.true_divide,
    np.negative,
    np.reciprocal,
    np.power,
    np.square,
    np.log,
]


class UfuncMixin(np.ndarray):
    """
    A mixin class that provides the overridding numpy ufunc functionality.
    """

    def _view_input_gf_as_ndarray(self, inputs, kwargs, meta):
        # View all input operands as np.ndarray to avoid infinite recursion
        v_inputs = list(inputs)
        # for i in meta["operands"]:
        for i in meta["gf_operands"]:
            if isinstance(inputs[i], self.__class__):
                v_inputs[i] = inputs[i].view(np.ndarray)

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if isinstance(output, self.__class__):
                    o = output.view(np.ndarray)
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    def _view_input_int_as_ndarray(self, inputs, meta):  # pylint: disable=no-self-use
        v_inputs = list(inputs)
        for i in meta["operands"]:
            if isinstance(inputs[i], int):
                # Use the largest valid dtype for this field
                v_inputs[i] = np.array(inputs[i], dtype=type(self).dtypes[-1])

        return v_inputs

    def _view_output_ndarray_as_gf(self, ufunc, v_outputs):
        if v_outputs is NotImplemented:
            return v_outputs
        if ufunc.nout == 1:
            v_outputs = (v_outputs, )

        outputs = []
        for v_output in v_outputs:
            o = self.__class__(v_output, dtype=self.dtype)
            outputs.append(o)

        return outputs[0] if len(outputs) == 1 else outputs

    def _verify_inputs(self, ufunc, method, inputs, meta):  # pylint: disable=too-many-branches
        types = [meta["types"][i] for i in meta["operands"]]  # List of types of the "operands", excludes index lists, etc
        operands = [inputs[i] for i in meta["operands"]]

        if method == "reduceat":
            return

        # Verify input operand types
        if ufunc in [np.add, np.subtract, np.true_divide, np.floor_divide]:
            if not all(t is self.__class__ for t in types):
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the same field {self.__class__}, not {types}")
        if ufunc in [np.multiply, np.power, np.square]:
            if not all(np.issubdtype(o.dtype, np.integer) or o.dtype == np.object_ for o in operands):
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the field {self.__class__} or integers, not {types}")
        if ufunc in [np.power, np.square]:
            if not types[0] is self.__class__:
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields can only exponentiate elements in the same field {self.__class__}, not {types[0]}")

        # Verify no divide by zero or log(0) errors
        if ufunc in [np.true_divide, np.floor_divide] and np.count_nonzero(operands[-1]) != operands[-1].size:
            raise ZeroDivisionError("Divide by 0")
        if ufunc is np.power:
            if method == "outer" and (np.any(operands[0] == 0) and np.any(operands[1] < 0)):
                raise ZeroDivisionError("Divide by 0")
            if method == "__call__" and np.any(np.logical_and(operands[0] == 0, operands[1] < 0)):
                raise ZeroDivisionError("Divide by 0")
        if ufunc is np.log and np.count_nonzero(operands[0]) != operands[0].size:
            raise ArithmeticError("Log(0) error")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # pylint: disable=too-many-branches
        """
        Intercept various numpy ufuncs (triggered by operators like `+` , `-`, etc). Then determine
        which operations will result in the correct answer in the given Galois field. Wherever
        appropriate, use native numpy ufuncs for their efficiency and generality in supporting various array
        shapes, etc.
        """
        meta = {}
        meta["types"] = [type(inputs[i]) for i in range(len(inputs))]
        meta["operands"] = list(range(0, len(inputs)))
        if method in ["at", "reduceat"]:
            # Remove the second argument for "at" ufuncs which is the indices list
            meta["operands"].pop(1)
        meta["gf_operands"] = [i for i in meta["operands"] if isinstance(inputs[i], self.__class__)]
        meta["non_gf_operands"] = [i for i in meta["operands"] if not isinstance(inputs[i], self.__class__)]

        # View Galois field array inputs as np.ndarray so subsequent numpy ufunc calls go to numpy and don't
        # result in infinite recursion
        inputs, kwargs = self._view_input_gf_as_ndarray(inputs, kwargs, meta)

        # For ufuncs we are not overriding, call the parent implementation
        if ufunc not in OVERRIDDEN_UFUNCS:
            return super().__array_ufunc__(ufunc, method, *inputs)  # pylint: disable=no-member

        inputs = self._view_input_int_as_ndarray(inputs, meta)

        self._verify_inputs(ufunc, method, inputs, meta)

        # Set all ufuncs with "casting" keyword argument to "unsafe" so we can cast unsigned integers
        # to integers. We know this is safe because we already verified the inputs.
        if method not in ["reduce", "accumulate", "at", "reduceat"]:
            kwargs["casting"] = "unsafe"

        # Need to set the intermediate dtype for reduction operations or an error will be thrown. We
        # use the largest valid dtype for this field.
        if method in ["reduce"]:
            kwargs["dtype"] = type(self).dtypes[-1]

        # Call appropriate ufunc method (implemented in subclasses)
        if ufunc is np.add:
            outputs = getattr(self._ufunc_add, method)(*inputs, **kwargs)
        elif ufunc is np.subtract:
            outputs = getattr(self._ufunc_subtract, method)(*inputs, **kwargs)
        elif ufunc is np.multiply:
            if meta["gf_operands"] == meta["operands"]:
                # In-field multiplication
                outputs = getattr(self._ufunc_multiply, method)(*inputs, **kwargs)
            else:
                # In-field "multiple addition" by an integer, i.e. GF(x) * 3 = GF(x) + GF(x) + GF(x)
                if 0 not in meta["gf_operands"]:
                    # If the integer is the first argument and the field element is the second, switch them. This
                    # is done because the ufunc needs to know which input is not in the field (so it can perform a
                    # modulus operation).
                    i = meta["gf_operands"][0]
                    j = meta["non_gf_operands"][0]
                    inputs[j], inputs[i] = inputs[i], inputs[j]
                outputs = getattr(self._ufunc_multiple_add, method)(*inputs, **kwargs)
        elif ufunc in [np.true_divide, np.floor_divide]:
            outputs = getattr(self._ufunc_divide, method)(*inputs, **kwargs)
        elif ufunc is np.negative:
            outputs = getattr(self._ufunc_negative, method)(*inputs, **kwargs)
        elif ufunc is np.reciprocal:
            outputs = getattr(self._ufunc_reciprocal, method)(*inputs, **kwargs)
        elif ufunc is np.power:
            outputs = getattr(self._ufunc_power, method)(*inputs, **kwargs)
        elif ufunc is np.square:
            inputs.append(np.array(2, dtype=self.dtype))
            outputs = getattr(self._ufunc_power, method)(*inputs, **kwargs)
        elif ufunc is np.log:
            outputs = getattr(self._ufunc_log, method)(*inputs, **kwargs)
        else:
            raise RuntimeError(f"The numpy ufunc '{ufunc.__name__}' wasn't processed even though it was supposed to be. Please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        if outputs is None or ufunc is np.log:
            return outputs
        else:
            outputs = self._view_output_ndarray_as_gf(ufunc, outputs)
            return outputs
