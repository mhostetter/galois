import numba
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
        return "<Galois Field: GF({}^{}), prim_poly = {} ({} decimal)>".format(cls.characteristic, cls.power, cls.prim_poly.str, None)


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
    _MUL_INV = None

    _numba_ufunc_add = None
    _numba_ufunc_subtract = None
    _numba_ufunc_multiply = None
    _numba_ufunc_divide = None
    _numba_ufunc_negative = None
    _numba_ufunc_power = None
    _numba_ufunc_log = None
    _numba_ufunc_poly_eval = None

    def __new__(cls, array, dtype=np.int64):
        if cls is GFBase:
            raise NotImplementedError("GFBase is an abstract base class that should not be directly instantiated")
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")

        array = cls._verify_and_convert(array, dtype=dtype)

        return array

    @classmethod
    def Zeros(cls, shape, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        return cls(np.zeros(shape, dtype=dtype), dtype=dtype)

    @classmethod
    def Ones(cls, shape, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        return cls(np.ones(shape, dtype=dtype), dtype=dtype)

    @classmethod
    def Random(cls, shape=(), low=0, high=None, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        if high is None:
            high = cls.order
        assert 0 <= low < cls.order and low < high <= cls.order
        return cls(np.random.randint(low, high, shape, dtype=dtype), dtype=dtype)

    @classmethod
    def Elements(cls, dtype=np.int64):
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.power}) arrays only support dtypes {cls.dtypes}, not {dtype}")
        return cls(np.arange(0, cls.order, dtype=dtype), dtype=dtype)

    def __str__(self):
        return self.__repr__()

    # def to_int_repr():

    # def to_poly_repr():

    # def to_log_repr():

    @classmethod
    def _export_globals(cls):
        pass

    @classmethod
    def target(cls, target):
        """
        Retarget the just-in-time compiled numba ufuncs.

        Parameters
        ----------
        target : str
            Either "cpu", "parallel", or "cuda".
        """
        if target not in ["cpu", "parallel", "cuda"]:
            raise ValueError(f"Valid numba compilation targets are [\"cpu\", \"parallel\", \"cuda\"], not {target}")

        cls._export_globals()

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
        cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(cls._add)
        cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(cls._subtract)
        cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(cls._multiply)
        cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(cls._divide)
        cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(cls._negative)
        cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(cls._power)
        cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(cls._log)
        cls._numba_ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(cls._poly_eval)

    @classmethod
    def _verify_and_convert(cls, array, dtype=np.int64):
        """
        Convert the input array into a Galois field array and check the input for data type and data range.
        """
        assert dtype in cls.dtypes

        # Convert the array-like object to a numpy array without specifying the desired dtype. This allows
        # numpy to determine the data type of the input array or list. This allows for detection of floating-point
        # inputs. We will convert to the desired dtype after checking that the input array are integers and
        # within the field. Use copy=True to prevent newly created array from sharing memory with input array.
        array = np.array(array, copy=True)
        assert np.issubdtype(array.dtype, np.integer), "Galois field elements must be integers not {}".format(array.dtype)
        assert np.all(array >= 0) and np.all(array < cls.order), "Galois field arrays must have elements with values less than the field order of {}".format(cls.order)

        # Convert array (already determined to be integers) to the Galois field's unsigned int dtype
        array = array.astype(dtype)
        array =  array.view(cls)

        return array

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        if dtype not in self.dtypes:
            raise TypeError(f"Galois field arrays can only be cast as integer dtypes {self.dtypes}, not {dtype}")
        return super().astype(dtype, **kwargs)

    def __array_finalize__(self, obj):
        """
        A numpy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype.
        """
        if obj is not None and not isinstance(obj, GFBase):
            if obj.dtype not in self.dtypes:
                raise TypeError(f"Galois field arrays can only have integer dtypes {self.dtypes}, not {obj.dtype}")
            if np.any(obj < 0) or np.any(obj >= self.order):
                raise ValueError(f"GF({self.order}) arrays must have values in [0, {self.order})")

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if np.isscalar(item):
            # Return scalar array elements are 0-dim Galois field arrays. This enables Galois field arithmetic
            # on scalars, which would otherwise be implemented using standard integer arithmetic.
            item = self.__class__(item)
        return item

    def __setitem__(self, key, value):
        # Verify the values to be written to the Galois field array are in the field
        self._verify_and_convert(value)
        super().__setitem__(key, value)

    def _view_input_gf_as_ndarray(self, inputs, kwargs):
        meta = {"types": [], "gf_inputs": [], "non_gf_inputs": []}

        # View all input arrays as np.ndarray to avoid infinite recursion
        v_inputs = []
        for i in range(len(inputs)):
            meta["types"].append(type(inputs[i]))
            if isinstance(inputs[i], self.__class__):
                meta["gf_inputs"].append(i)
                v_inputs.append(inputs[i].view(np.ndarray))
            else:
                meta["non_gf_inputs"].append(i)
                v_inputs.append(inputs[i])

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

        return v_inputs, kwargs, meta

    def _view_input_int_as_ndarray(self, inputs):  # pylint: disable=no-self-use
        v_inputs = []
        for input_ in inputs:
            if isinstance(input_, int):
                # i = np.array(input_, dtype=self._dtype)
                i = np.array(input_)
            else:
                i = input_
            v_inputs.append(i)

        return v_inputs

    def _view_output_ndarray_as_gf(self, ufunc, v_outputs):
        if v_outputs is NotImplemented:
            return v_outputs
        if ufunc.nout == 1:
            v_outputs = (v_outputs, )

        outputs = []
        for v_output in v_outputs:
            o = self.__class__(v_output)
            outputs.append(o)

        return outputs[0] if len(outputs) == 1 else outputs

    def _verify_inputs(self, ufunc, method, inputs, meta):  # pylint: disable=too-many-branches
        if method == "reduceat":
            return

        for i in meta["non_gf_inputs"]:
            if method == "at" and i == 1:
                continue
            if ufunc in [np.add, np.subtract, np.true_divide, np.floor_divide]:
                if not np.issubdtype(inputs[i].dtype, np.integer):
                    raise TypeError(f"Operation \"{ufunc.__name__}\" in Galois field must be performed on integers not {inputs[i].dtype}")
                if np.any(inputs[i] < 0) or np.all(inputs[i] >= self.order):
                    raise ValueError(f"Operation \"{ufunc.__name__}\" in Galois field must be performed with elements in the field [0, {self.order})")
            elif ufunc in [np.multiply, np.power, np.square]:
                if not np.issubdtype(inputs[i].dtype, np.integer):
                    raise TypeError(f"Operation \"{ufunc.__name__}\" in Galois field must be performed with elements in Z, the integers")

        if ufunc in [np.true_divide, np.floor_divide] and np.count_nonzero(inputs[-1]) != inputs[-1].size:
            raise ZeroDivisionError("Divide by 0")
        if ufunc is np.power:
            if method == "outer" and (np.any(inputs[0] == 0) and np.any(inputs[1] < 0)):
                raise ZeroDivisionError("Divide by 0")
            if method == "__call__" and np.any(np.logical_and(inputs[0] == 0, inputs[1] < 0)):
                raise ZeroDivisionError("Divide by 0")
        if ufunc is np.log and np.count_nonzero(inputs[0]) != inputs[0].size:
            raise ArithmeticError("Log(0) error")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept various numpy ufuncs (triggered by operators like `+` , `-`, etc). Then determine
        which operations will result in the correct answer in the given Galois field. Wherever
        appropriate, use native numpy ufuncs for their efficiency and generality in supporting various array
        shapes, etc.
        """
        # View Galois field array inputs as np.ndarray so subsequent numpy ufunc calls go to numpy and don't
        # result in infinite recursion
        inputs, kwargs, meta = self._view_input_gf_as_ndarray(inputs, kwargs)

        # For ufuncs we are not overriding, call the parent implementation
        if ufunc not in OVERRIDDEN_UFUNCS.keys():
            return super().__array_ufunc__(ufunc, method, *inputs)  # pylint: disable=no-member

        inputs = self._view_input_int_as_ndarray(inputs)

        self._verify_inputs(ufunc, method, inputs, meta)

        # if method == "__call__":
        if method not in ["reduce", "accumulate", "at", "reduceat"]:
            kwargs["casting"] = "unsafe"

        # Call appropriate ufunc method (implemented in subclasses)
        if ufunc is np.add:
            outputs = getattr(self._numba_ufunc_add, method)(*inputs, **kwargs)
        elif ufunc is np.subtract:
            outputs = getattr(self._numba_ufunc_subtract, method)(*inputs, **kwargs)
        elif ufunc is np.multiply:
            outputs = getattr(self._numba_ufunc_multiply, method)(*inputs, **kwargs)
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

    @staticmethod
    def _add(a, b):
        raise NotImplementedError

    @staticmethod
    def _subtract(a, b):
        raise NotImplementedError

    @staticmethod
    def _multiply(a, b):
        raise NotImplementedError

    @staticmethod
    def _divide(a, b):
        raise NotImplementedError

    @staticmethod
    def _negative(a):
        raise NotImplementedError

    @staticmethod
    def _power(a, b):
        raise NotImplementedError

    @staticmethod
    def _log(a):
        raise NotImplementedError

    @staticmethod
    def _poly_eval(coeffs, values, results):
        raise NotImplementedError
