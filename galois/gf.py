import numpy as np


# List of numpy ufuncs that Galois field arrays will override. All other ufuncs should
# be handled by numpy normally.
OVERRIDDEN_UFUNCS = [np.add, np.subtract, np.multiply, np.true_divide, np.floor_divide, np.negative, np.square, np.power, np.log]

# List of numpy ufuncs that Galois field arrays will use to perform field arithmetic. This
# list is needed so we perform "valid field element" checks only on these ufuncs.
IN_FIELD_UFUNCS = [np.add, np.subtract, np.multiply, np.true_divide, np.floor_divide]


def _determine_dtype(order):
    # Determine the minimum unsigned int dtype to store the field elements
    bits = int(np.ceil(np.log2(order)))
    assert 1 <= bits <= 16, "Currently only field orders of 2 to 65536 are supported, submit a GitHub issue (https://github.com/mhostetter/galois/issues) if larger fields are desired"
    if bits <= 8:
        dtype = np.uint8
    else:
        dtype = np.uint16
    return dtype


class _GF(np.ndarray):
    """
    asdf

    .. code-block:: python

        import galois
        print(galois.GF2)
        print(galois.GF2.Elements())
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
    gf.Poly: The primitive polynomial of the Galois field `GF(p^m)`. The primitve polynomial must have coefficients
    in `GF(p)`.
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
    def Random(cls, shape=()):
        return cls(np.random.randint(0, cls.order, shape, dtype=cls._dtype))

    @classmethod
    def _verify_and_convert(cls, array):
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
        if obj is not None and not isinstance(obj, _GF):
            # Invoked during view casting
            assert obj.dtype == self._dtype, "Can only view cast to Galois field arrays if the input array has the same dtype of {}".format(self._dtype)
            assert np.all(obj >= 0) and np.all(obj < self.order), "Galois field arrays must have elements with values less than the field order of {}".format(self.order)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if np.isscalar(item):
            # Return scalar array elements are 0-dim Galois field arrays. This enables Galois field arithmetic
            # on scalars, which would otherwise be implemented using standard integer arithmetic.
            item = self.__class__(item)
        return item

    def __setitem__(self, key, value):
        # Verify the values to be written to the Galois field array are within the field
        self._verify_and_convert(value)
        super().__setitem__(key, value)

    def _view_ufunc_gf_as_ndarray(self, inputs, kwargs):
        # View all input arrays as np.ndarray to avoid infinite recursion
        v_inputs = []
        for input_ in inputs:
            if isinstance(input_, self.__class__):
                i = input_.view(np.ndarray)
            else:
                i = input_
            v_inputs.append(i)

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if isinstance(output, (int, self.__class__)):
                    v_outputs.append(output.view(np.ndarray))
                else:
                    v_outputs.append(output)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    def _verify_ufunc_input_range(self, ufunc, inputs):
        if ufunc not in IN_FIELD_UFUNCS:
            return
        for input_ in inputs:
            self._verify_and_convert(input_)

    def _view_ufunc_int_as_ndarray(self, inputs):
        v_inputs = []
        for input_ in inputs:
            if isinstance(input_, int):
                i = np.array(input_, dtype=self._dtype)
            else:
                i = input_
            v_inputs.append(i)

        return v_inputs

    def _view_ufunc_ndarray_as_gf(self, ufunc, v_outputs):
        if v_outputs is NotImplemented:
            return v_outputs
        if ufunc.nout == 1:
            v_outputs = (v_outputs, )

        outputs = []
        for v_output in v_outputs:
            if isinstance(v_output, np.ndarray):
                o = v_output.view(self.__class__)
            else:
                o = self.__class__(v_output)
            outputs.append(o)

        return outputs[0] if len(outputs) == 1 else outputs
