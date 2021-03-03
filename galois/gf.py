import numpy as np


UFUNC_MAP = {
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


class _GFMeta(type):
    """
    Defines a metaclass to give all GF classes a __str__() special method, not just their instances.
    """

    def __str__(cls):
        return "<Galois Field: GF({}^{}), prim_poly = {} ({} decimal)>".format(cls.characteristic, cls.power, cls.prim_poly.str, None)


class _GF(np.ndarray, metaclass=_GFMeta):
    """
    asdf
    """

    # NOTE: These class attributes will be set in the subclasses of _GF

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

    _dtype = None

    def __new__(cls, array):
        assert cls is not _GF, "_GF is an abstract base class that should not be directly instantiated"
        array = cls._verify_and_convert(array)
        return array

    @classmethod
    def Zeros(cls, shape):
        return cls(np.zeros(shape, dtype=cls._dtype))

    @classmethod
    def Ones(cls, shape):
        return cls(np.ones(shape, dtype=cls._dtype))

    @classmethod
    def Random(cls, shape=(), low=0, high=None):
        if high is None:
            high = cls.order
        assert 0 <= low < cls.order and low < high <= cls.order
        return cls(np.random.randint(low, high, shape, dtype=cls._dtype))

    @classmethod
    def Elements(cls):
        return cls(np.arange(0, cls.order, dtype=cls._dtype))

    # def to_int_repr():

    # def to_poly_repr():

    # def to_log_repr():

    @classmethod
    def _verify_and_convert(cls, array):
        """
        Convert the input array into a Galois field array and check the input for data type and data range.
        """
        # Convert the array-like object to a numpy array without specifying the desired dtype. This allows
        # numpy to determine the data type of the input array or list. This allows for detection of floating-point
        # inputs. We will convert to the desired dtype after checking that the input array are integers and
        # within the field. Use copy=True to prevent newly created array from sharing memory with input array.
        array = np.array(array, copy=True)
        assert np.issubdtype(array.dtype, np.integer), "Galois field elements must be integers not {}".format(array.dtype)
        assert np.all(array >= 0) and np.all(array < cls.order), "Galois field arrays must have elements with values less than the field order of {}".format(cls.order)

        # Convert array (already determined to be integers) to the Galois field's unsigned int dtype
        array = array.astype(cls._dtype)
        array =  array.view(cls)

        return array

    def __array_finalize__(self, obj):
        """
        A numpy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype.
        """
        if obj is not None and not isinstance(obj, _GF):
            # Invoked during view casting
            assert obj.dtype == self._dtype, "Can only view cast to Galois field arrays if the input array has the field's dtype of {}".format(self._dtype)
            assert np.all(obj >= 0) and np.all(obj < self.order), "Galois field arrays must have elements with values less than the field order of {}".format(self.order)

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

    def _ufunc_verify_input_in_field(self, ufunc, array):
        assert np.issubdtype(array.dtype, np.integer) and np.all(array >= 0) and np.all(array < self.order), "Operation \"{}\" in Galois field must be performed with elements in the field [0, {})".format(ufunc.__name__, self.order)

    def _ufunc_verify_input_in_positive_integers(self, ufunc, array):  # pylint: disable=no-self-use
        assert np.issubdtype(array.dtype, np.integer) and np.all(array >= 0), "Operation \"{}\" in Galois field must be performed with elements in N, the natural numbers".format(ufunc.__name__)

    def _ufunc_verify_input_in_integers(self, ufunc, array):  # pylint: disable=no-self-use
        assert np.issubdtype(array.dtype, np.integer), "Operation \"{}\" in Galois field must be performed with elements in Z, the integers".format(ufunc.__name__)

    def _ufunc_view_input_gf_as_ndarray(self, inputs, kwargs):
        meta = {"types": [], "gf_inputs": []}

        # View all input arrays as np.ndarray to avoid infinite recursion
        v_inputs = []
        for i in range(len(inputs)):
            meta["types"].append(type(inputs[i]))
            if isinstance(inputs[i], self.__class__):
                meta["gf_inputs"].append(i)
                v_inputs.append(inputs[i].view(np.ndarray))
            else:
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

    def _ufunc_view_input_int_as_ndarray(self, inputs):  # pylint: disable=no-self-use
        v_inputs = []
        for input_ in inputs:
            if isinstance(input_, int):
                # i = np.array(input_, dtype=self._dtype)
                i = np.array(input_)
            else:
                i = input_
            v_inputs.append(i)

        return v_inputs

    def _ufunc_view_output_ndarray_as_gf(self, ufunc, v_outputs):
        if v_outputs is NotImplemented:
            return v_outputs
        if ufunc.nout == 1:
            v_outputs = (v_outputs, )

        outputs = []
        for v_output in v_outputs:
            o = self.__class__(v_output)
            outputs.append(o)

        return outputs[0] if len(outputs) == 1 else outputs

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept various numpy ufuncs (triggered by operators like `+` , `-`, etc). Then determine
        which operations will result in the correct answer in the given Galois field. Wherever
        appropriate, use native numpy ufuncs for their efficiency and generality in supporting various array
        shapes, etc.
        """
        # print(ufunc, method, inputs)
        inputs, kwargs, meta = self._ufunc_view_input_gf_as_ndarray(inputs, kwargs)

        if ufunc not in UFUNC_MAP.keys():
            return super().__array_ufunc__(ufunc, method, *inputs)  # pylint: disable=no-member

        inputs = self._ufunc_view_input_int_as_ndarray(inputs)

        # Call appropriate ufunc method (implemented in subclasses)
        outputs = getattr(self, UFUNC_MAP[ufunc])(ufunc, method, inputs, kwargs, meta)

        if ufunc is np.log:
            return outputs
        else:
            outputs = self._ufunc_view_output_ndarray_as_gf(ufunc, outputs)
            return outputs

    def _add(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError

    def _subtract(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError

    def _multiply(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError

    def _divide(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError

    def _negative(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError

    def _power(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError

    def _square(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError

    def _log(self, ufunc, method, inputs, kwargs, meta):
        raise NotImplementedError
