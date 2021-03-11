import numpy as np


OVERRIDDEN_UFUNCS = {
    np.add: "_add",
    np.subtract: "_subtract",
    np.multiply: "_multiply",
    np.floor_divide: "_divide",
    np.true_divide: "_divide",
    np.negative: "_negative",
    np.power: "_power",
    np.square: "_square",
    np.log: "_log"
}

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]


class GFBaseMeta(type):
    """
    Defines a metaclass to give all GF classes a __str__() special method, not just their instances.
    """

    def __str__(cls):
        return "<Galois Field: GF({}^{}), prim_poly = {} ({} decimal)>".format(cls.characteristic, cls.power, cls.prim_poly.str, cls.prim_poly.decimal)


class GFBase(np.ndarray, metaclass=GFBaseMeta):
    """
    asdf

    .. note::
        This is an abstract base class for all Galois fields. It cannot be instantiated directly.
    """

    # NOTE: These class attributes will be set in the subclasses of GFBase

    characteristic = None
    """
    int: The characteristic `p`, which must be prime, of the Galois field `GF(p^m)`. Adding `p` copies of any
    element will always result in 0.
    """

    power = None
    """
    int: The power `m`, which must be non-negative, of the Galois field `GF(p^m)`.
    """

    order = None
    """
    int: The order `p^m` of the Galois field `GF(p^m)`. The order of the field is also equal to the field's size.
    """

    prim_poly = None
    """
    galois.Poly: The primitive polynomial of the Galois field `GF(p^m)`. The primitve polynomial must have coefficients
    in `GF(p)`.
    """

    alpha = None
    """
    int: The primitive element of the Galois field `GF(p^m)`. The primitive element is a root of the primitive polynomial,
    such that `prim_poly(alpha) = 0`. The primitive element also generates the field `GF(p^m) = {0, 1, alpha^1, alpha*2,
    ..., alpha^(p^m - 2)}`.
    """

    dtypes = []
    """
    list: List of valid integer numpy dtypes that are compatible with this Galois field array class.
    """

    _EXP = None
    _LOG = None

    _numba_ufunc_add = None
    _numba_ufunc_subtract = None
    _numba_ufunc_multiply = None
    _numba_ufunc_divide = None
    _numba_ufunc_negative = None
    _numba_ufunc_multiple_add = None
    _numba_ufunc_power = None
    _numba_ufunc_log = None
    _numba_ufunc_poly_eval = None

    def __new__(cls, array, dtype=np.int64):
        if cls is GFBase:
            raise NotImplementedError("GFBase is an abstract base class that cannot be directly instantiated")
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")

        # Convert the array-like object to a numpy array without specifying the desired dtype. This allows
        # numpy to determine the data type of the input array, list, tuple, etc. This allows for detection of
        # floating-point inputs. We will convert to the desired dtype after checking that the input array are integers
        # and within the field. We use `copy=True` to prevent newly created array from sharing memory with input array.
        array = np.array(array, copy=True)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"Galois field array elements must have integer dtypes, not {array.dtype}")
        if np.any(array < 0) or np.any(array >= cls.order):
            raise ValueError(f"Galois field arrays must have elements in [0, {cls.order}), not {array}")

        # Convert array (already determined to be integers) to the Galois field's unsigned int dtype
        array = array.astype(dtype)
        array =  array.view(cls)

        return array

    @classmethod
    def Zeros(cls, shape, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        return np.zeros(shape, dtype=dtype).view(cls)

    @classmethod
    def Ones(cls, shape, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        return np.ones(shape, dtype=dtype).view(cls)

    @classmethod
    def Random(cls, shape=(), low=0, high=None, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        if high is None:
            high = cls.order
        assert 0 <= low < cls.order and low < high <= cls.order
        return np.random.randint(low, high, shape, dtype=dtype).view(cls)

    @classmethod
    def Elements(cls, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        return np.arange(0, cls.order, dtype=dtype).view(cls)

    def __str__(self):
        return self.__repr__()

    # def to_int_repr():

    # def to_poly_repr():

    # def to_log_repr():

    @classmethod
    def target(cls, target):
        """
        Retarget the just-in-time compiled numba ufuncs.

        Parameters
        ----------
        target : str
            Either "cpu", "parallel", or "cuda".
        """
        raise NotImplementedError

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        if dtype not in self.dtypes:
            raise TypeError(f"Galois field arrays can only be cast as integer dtypes {self.dtypes}, not {dtype}")
        return super().astype(dtype, **kwargs)

    def __array_finalize__(self, obj):
        """
        A numpy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype and that the values are in the field.
        """
        if obj is not None and not isinstance(obj, GFBase):
            if obj.dtype not in self.dtypes:
                raise TypeError(f"Galois field arrays can only have integer dtypes {self.dtypes}, not {obj.dtype}")
            if np.any(obj < 0) or np.any(obj >= self.order):
                raise ValueError(f"GF({self.order}) arrays must have values in [0, {self.order})")

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if np.isscalar(item):
            # Return scalar array elements as 0-dimension Galois field arrays. This enables Galois field arithmetic
            # on scalars, which would otherwise be implemented using standard integer arithmetic.
            item = self.__class__(item, dtype=self.dtype)
        return item

    def __setitem__(self, key, value):
        # Verify the values to be written to the Galois field array are in the field
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"Galois field array elements must have integer dtypes, not {array.dtype}")
        if np.any(array < 0) or np.any(array >= self.order):
            raise ValueError(f"Galois field arrays must have elements in [0, {self.order}), not {array}")
        super().__setitem__(key, value)

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
                v_inputs[i] = np.array(inputs[i], dtype=np.int64)

        return v_inputs
        # return inputs, meta

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
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the same field {repr(self.__class__)}, not {types}")
        if ufunc in [np.multiply, np.power, np.square]:
            if not all(np.issubdtype(o.dtype, np.integer) for o in operands):
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the field {repr(self.__class__)} or integers, not {types}")
        if ufunc in [np.power, np.square]:
            if not types[0] is self.__class__:
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields can only exponentiate elements in the same field {repr(self.__class__)}, not {types[0]}")

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
        if ufunc not in OVERRIDDEN_UFUNCS.keys():
            return super().__array_ufunc__(ufunc, method, *inputs)  # pylint: disable=no-member

        inputs = self._view_input_int_as_ndarray(inputs, meta)

        self._verify_inputs(ufunc, method, inputs, meta)

        # Set all ufuncs with "casting" keyword argument to "unsafe" so we can cast unsigned integers
        # to integers. We know this is safe because we already verified the inputs.
        if method not in ["reduce", "accumulate", "at", "reduceat"]:
            kwargs["casting"] = "unsafe"

        # Call appropriate ufunc method (implemented in subclasses)
        if ufunc is np.add:
            outputs = getattr(self._numba_ufunc_add, method)(*inputs, **kwargs)
        elif ufunc is np.subtract:
            outputs = getattr(self._numba_ufunc_subtract, method)(*inputs, **kwargs)
        elif ufunc is np.multiply:
            if meta["gf_operands"] == meta["operands"]:
                # In-field multiplication
                outputs = getattr(self._numba_ufunc_multiply, method)(*inputs, **kwargs)
            else:
                # In-field "multiple addition" by an integer, i.e. GF(x) * 3 = GF(x) + GF(x) + GF(x)
                if 0 not in meta["gf_operands"]:
                    # If the integer is the first argument and the field element is the second, switch them. This
                    # is done because the ufunc needs to know which input is not in the field (so it can perform a
                    # modulus operation).
                    i = meta["gf_operands"][0]
                    j = meta["non_gf_operands"][0]
                    inputs[j], inputs[i] = inputs[i], inputs[j]
                outputs = getattr(self._numba_ufunc_multiple_add, method)(*inputs, **kwargs)
        elif ufunc in [np.true_divide, np.floor_divide]:
            outputs = getattr(self._numba_ufunc_divide, method)(*inputs, **kwargs)
        elif ufunc is np.negative:
            outputs = getattr(self._numba_ufunc_negative, method)(*inputs, **kwargs)
        elif ufunc is np.power:
            outputs = getattr(self._numba_ufunc_power, method)(*inputs, **kwargs)
        elif ufunc is np.square:
            inputs.append(np.array([2], dtype=self.dtype))
            outputs = getattr(self._numba_ufunc_power, method)(*inputs, **kwargs)
        elif ufunc is np.log:
            outputs = getattr(self._numba_ufunc_log, method)(*inputs, **kwargs)

        if outputs is None or ufunc is np.log:
            return outputs
        else:
            outputs = self._view_output_ndarray_as_gf(ufunc, outputs)
            return outputs
