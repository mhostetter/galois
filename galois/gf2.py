import numba
import numpy as np

from .gf import _GF


@numba.vectorize(["uint8(uint8, uint8)", "uint8(uint8, int64)"], nopython=True)
def _numba_ufunc_power(a, b):
    # Calculate a**b
    if a == 0:
        return 0
    elif b == 0:
        return 1
    else:
        return a


class GF2(_GF):
    """
    asdf

    Examples
    --------

    GF2 class properties

    .. ipython:: python

        print(galois.GF2)
        galois.GF2.characteristic
        galois.GF2.power
        galois.GF2.order
        galois.GF2.prim_poly

    Construct arrays in GF2

    .. ipython:: python

        a = galois.GF2([1,0,1,1]); a
        b = galois.GF2([1,1,1,1]); b

    Arithmetic with GF2 arrays

    .. ipython:: python

        # Element-wise addition
        a + b
        # Element-wise subtraction
        a - b
        # Element-wise multiplication
        a * b
        # Element-wise division
        a / b
    """

    characteristic = 2
    power = 1
    order = 2
    prim_poly = None
    alpha = 1
    _dtype = np.uint8

    def _add(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(2), 0+1=1 and 1+1=0, which is equivalent to bitwise XOR. Subtraction is the same as addition
        in GF(2) because each element is its additive inverse.
        """
        self._ufunc_verify_input_in_field(ufunc, inputs[0])
        self._ufunc_verify_input_in_field(ufunc, inputs[1])
        outputs = getattr(np.bitwise_xor, method)(*inputs, **kwargs)
        return outputs

    def _subtract(self, ufunc, method, inputs, kwargs, meta):
        return self._add(ufunc, method, inputs, kwargs, meta)

    def _multiply(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(2), 0*a=0 for any a >= 0 and 1*a=1 for any a > 0, which is equivalent to bitwise AND.
        """
        for i in [0,1]:
            if i in meta["gf_inputs"]:
                self._ufunc_verify_input_in_field(ufunc, inputs[i])
            else:
                self._ufunc_verify_input_in_integers(ufunc, inputs[i])
        outputs = getattr(np.bitwise_and, method)(*inputs, casting="unsafe", **kwargs)
        return outputs

    def _divide(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(2), 0/1=0 and 1/1=1, which is equivalent to bitwise AND.
        """
        assert np.count_nonzero(inputs[1]) == inputs[1].size, "Cannont divide by 0 in a Galois field"
        self._ufunc_verify_input_in_field(ufunc, inputs[0])
        self._ufunc_verify_input_in_field(ufunc, inputs[1])
        # if np.count_nonzero(inputs[1]) != inputs[1].size:
        #     warnings.warn("divide by zero encountered in \"{}\", 0 is outputted where 'Inf' would otherwise".format(ufunc), RuntimeWarning)
        outputs = getattr(np.bitwise_and, method)(*inputs, casting="unsafe", **kwargs)
        return outputs

    def _negative(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(2), each element is its additive inverse.
        """
        outputs = inputs[0]
        return outputs

    def _power(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(2), a^0=1 and a^b=a for any a,b in GF(2). We created a special numba-optimized ufunc to perform
        this operation on all elements in the two input arrays a,b.
        """
        assert not np.any(np.logical_and(inputs[0] == 0, inputs[1] < 0)), "Cannot exponentiate 0 to a negative power in a Galois field"
        self._ufunc_verify_input_in_field(ufunc, inputs[0])
        self._ufunc_verify_input_in_integers(ufunc, inputs[1])
        # if np.any(np.logical_and(inputs[0] == 0, inputs[1] < 0)):
        #     warnings.warn("divide by zero encountered in \"{}\", 0 is outputted where 'Inf' would otherwise be".format(ufunc), RuntimeWarning)
        inputs[1] = np.mod(inputs[1], self.order - 1)  # x^q = x, where q is the order
        inputs[1] = inputs[1].astype(self._dtype)
        outputs = getattr(_numba_ufunc_power, method)(*inputs, **kwargs)
        return outputs

    def _square(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(2), 0^2=0 and 1^2=1.
        """
        outputs = inputs[0]
        return outputs

    def _log(self, ufunc, method, inputs, kwargs, meta):
        """
        Log base alpha of elements in GF(2). In GF(2), log(1) = 0 and log(0) is undefined.
        """
        assert np.count_nonzero(inputs[0]) == inputs[0].size, "Cannont compute log(0) in a Galois field"
        outputs = getattr(np.multiply, method)(*inputs, 0, **kwargs)
        return outputs
