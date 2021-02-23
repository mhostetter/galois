import warnings
import numba
import numpy as np

from .gf import _GF, OVERRIDDEN_UFUNCS


@numba.vectorize("uint8(uint8, uint8)", nopython=True)
def _ufunc_power(a, b):
    # Calculate a**b
    if b == 0:
        return 1
    else:
        return a


@numba.vectorize("float32(uint8)", nopython=True)
def _ufunc_log_alpha(a):
    # Calculate log_alpha(a).
    if a == 0:
        return -np.Inf
    else:
        return 0


class GF2(_GF):
    """
    asdf
    """

    characteristic = 2
    power = 1
    order = 2
    prim_poly = None
    _dtype = np.uint8

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs, kwargs = self._view_ufunc_gf_as_ndarray(inputs, kwargs)

        if ufunc not in OVERRIDDEN_UFUNCS:
            return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member

        self._verify_ufunc_input_range(ufunc, inputs)

        inputs = self._view_ufunc_int_as_ndarray(inputs)

        # Intercept various numpy ufuncs (triggered by operators like `+` , `-`, etc). Then determine
        # which operations will result in the correct answer in the given Galois field. Wherever
        # appropriate, use native numpy ufuncs for their efficiency and generality in supporting various array
        # shapes, etc.

        if ufunc in [np.add, np.subtract]:
            # In GF2, 0+1=1 and 1+1=0, which is equivalent to bitwise XOR. Subtraction is the same as addition
            # in GF2 because each element is its additive inverse.
            outputs = getattr(np.bitwise_xor, method)(*inputs, **kwargs)

        elif ufunc is np.multiply:
            # In GF2, 0*1=0 and 1*1=1, which is equivalent to bitwise AND.
            outputs = getattr(np.bitwise_and, method)(*inputs, **kwargs)

        elif ufunc in [np.floor_divide, np.true_divide]:
            # In GF2, 0/1=0 and 1/1=1, which is equivalent to bitwise AND.
            if np.count_nonzero(inputs[1]) != inputs[1].size:
                warnings.warn("divide by zero encountered in _ufunc_divide, 0 is outputted where 'Inf' would otherwise be", RuntimeWarning)
            outputs = getattr(np.bitwise_and, method)(*inputs, **kwargs)

        elif ufunc is np.negative:
            # In GF2, each element is its additive inverse.
            outputs = inputs[0]

        elif ufunc is np.power:
            # In GF2, a^0=1 and a^b=a for any a,b in GF2. We created a special numba-optimized ufunc to perform
            # this operation on all elements in the two input arrays a,b.
            outputs = getattr(_ufunc_power, method)(*inputs)

        elif ufunc is np.square:
            outputs = getattr(_ufunc_power, method)(*inputs, 2)

        elif ufunc is np.log:
            # Log base alpha of elements in GF2. In GF2, log(0)=-Inf. This will eturn a regular (non-finite field)
            # integer in dtype=np.float32.
            output = getattr(_ufunc_log_alpha, method)(*inputs)
            return output.item() if output.ndim == 0 else output

        else:
            outputs = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member

        outputs = self._view_ufunc_ndarray_as_gf(ufunc, outputs)

        return outputs
