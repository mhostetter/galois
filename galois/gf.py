import numpy as np


class _GF(np.ndarray):
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
    gf.Poly: The primitive polynomial of the Galois field `GF(p^m)`. The primitve polynomial must have coefficients
    in `GF(p)`.
    """

    def __new__(cls, array):
        assert cls is not _GF, "_GF is an abstract base class that should not be directly instantiated"

        # Determine the minimum unsigned int dtype to store the field elements
        bits = int(np.ceil(np.log2(cls.order)))
        assert 1 <= bits <= 16
        if bits <= 8:
            dtype = np.uint8
        else:
            dtype = np.uint16

        # Use copy=True to prevent newly created array from sharing memory with input array
        array = np.array(array, dtype=dtype, copy=True)
        assert np.all(array < cls.order), "Elements of Galois field arrays must have elements with values less than the field order of {}".format(cls.order)

        return array.view(cls)

    @classmethod
    def Zeros(cls, shape):
        return cls(np.zeros(shape))

    @classmethod
    def Ones(cls, shape):
        return cls(np.ones(shape))

    @classmethod
    def Random(cls, shape=()):
        return cls(np.random.randint(0, cls.order, shape))

    # def __str__(self):
    #     return "Galois Field"

    # def __array_finalize__(self, obj):
    #     pass

    def _pre_ufunc(self, ufunc, method, inputs, kwargs):  # pylint: disable=unused-argument
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

    def _post_ufunc(self, ufunc, method, v_outputs):  # pylint: disable=unused-argument
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
