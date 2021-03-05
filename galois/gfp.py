import numba
import numpy as np

from .algorithm import is_prime, extended_euclidean_algorithm, primitive_roots, modular_exp, min_poly
from .gf import _GF


class GFp(_GF):
    """
    asdf
    """

    _MUL_INV = []
    _numba_ufunc_power = None
    _numba_ufunc_log = None

    def __new__(cls, *args, **kwargs):
        assert cls is not GFp, "GFp is an abstract base class, it cannot be instantiated directly; use GFp_factory() to generate a GF(p) class"
        return super().__new__(cls, *args, **kwargs)

    def _add(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(p), addition can be computed using addition in the integer field Z and the
        integer field result is taken `mod p` to get the result in GF(p).
        """
        self._ufunc_verify_input_in_field(ufunc, inputs[0])
        self._ufunc_verify_input_in_field(ufunc, inputs[1])
        output = getattr(np.add, method)(*inputs, **kwargs)
        output = getattr(np.mod, method)(output, self.order, **kwargs)
        return output

    def _subtract(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(p), subtraction can be computed using subtraction in the integer field Z and the
        integer field result is taken `mod p` to get the result in GF(p).
        """
        self._ufunc_verify_input_in_field(ufunc, inputs[0])
        self._ufunc_verify_input_in_field(ufunc, inputs[1])
        output = getattr(np.subtract, method)(*inputs, **kwargs)
        output = getattr(np.mod, method)(output, self.order, **kwargs)
        return output

    def _multiply(self, ufunc, method, inputs, kwargs, meta):
        """
        In GF(p), multiplication can be computed using multiplication in the integer field Z and the
        integer field result is taken `mod p` to get the result in GF(p).
        """
        for i in [0,1]:
            if i in meta["gf_inputs"]:
                self._ufunc_verify_input_in_field(ufunc, inputs[i])
            else:
                self._ufunc_verify_input_in_integers(ufunc, inputs[i])
        output = getattr(np.multiply, method)(*inputs, **kwargs)
        output = getattr(np.mod, method)(output, self.order, **kwargs)
        return output

    def _divide(self, ufunc, method, inputs, kwargs, meta):
        assert np.count_nonzero(inputs[1]) == inputs[1].size, "Cannont divide by 0 in a Galois field"
        self._ufunc_verify_input_in_field(ufunc, inputs[0])
        self._ufunc_verify_input_in_field(ufunc, inputs[1])
        # if np.count_nonzero(inputs[1]) != inputs[1].size:
        #     warnings.warn("divide by zero encountered in \"{}\", 0 is outputted where 'Inf' would otherwise".format(ufunc), RuntimeWarning)
        dividend = inputs[0]
        divisor = inputs[1]
        inv_divisor = self._MUL_INV[divisor]  # Lookup the multiplicative inverse of the divisor
        return self._multiply(ufunc, method, (dividend, inv_divisor), kwargs, meta)
        # outputs = getattr(np.bitwise_and, method)(*inputs, **kwargs)
        # return outputs

    def _negative(self, ufunc, method, inputs, kwargs, meta):
        output = getattr(np.multiply, method)(inputs[0], -1, **kwargs)
        output = getattr(np.mod, method)(output, self.order, **kwargs)
        return output

    def _power(self, ufunc, method, inputs, kwargs, meta):
        assert not np.any(np.logical_and(inputs[0] == 0, inputs[1] < 0)), "Cannot exponentiate 0 to a negative power in a Galois field"
        self._ufunc_verify_input_in_integers(ufunc, inputs[1])
        outputs = getattr(self._numba_ufunc_power, method)(*inputs, **kwargs)
        return outputs

    def _square(self, ufunc, method, inputs, kwargs, meta):
        inputs.append(np.array(2, dtype=int))
        return self._power(ufunc, method, inputs, kwargs, meta)

    def _log(self, ufunc, method, inputs, kwargs, meta):
        """
        Log base alpha of elements in GF(p).
        """
        assert np.count_nonzero(inputs[0]) == inputs[0].size, "Cannont compute log(0) in a Galois field"
        outputs = getattr(self._numba_ufunc_log, method)(*inputs, **kwargs)
        return outputs


def _ufunc_power(a, b):
    # Calculate a**b
    if b < 0:
        a = MUL_INV[a]
        b = abs(b)
    result = 1
    for _ in range(0, b):
        result = (result * a) % ORDER
    return result


def _ufunc_log(a):
    # Calculate np.log(a)
    return LOG[a]


def _build_luts(p, alpha, dtype):
    """
    Constructs the multiplicative inverse lookup table.

    Parameters
    ----------
    p : int
        Galois field prime characteristic.
    alpha : int
        The primitive element of the Galois field.
    dtype : np.dtype
        Numpy data type for lookup tables.

    Returns
    -------
    np.ndarray
        The anti-log lookup table for the field. `EXP[i] = alpha^i`.
    np.ndarray
        The log lookup table for the field. `LOG[i] = log_alpha(i)`.
    np.ndarray
        The multiplicative inverse lookup table for the field. `MUL_INV[i] = 1/i`.
    """
    assert isinstance(p, int)
    order = p

    # exp = (alpha ** np.arange(0, order, dtype=dtype)) % order
    exp = modular_exp(alpha, np.arange(0, order, dtype=dtype), order)
    log = np.zeros(order, dtype=dtype)
    mul_inv = np.zeros(order, dtype=dtype)

    log[0] = 0  # Technically -Inf
    mul_inv[0] = 0  # Technically -Inf
    for i in range(1, order):
        if i < order-1:
            log[exp[i]] = i
        x, _, _ = extended_euclidean_algorithm(i, order)
        x = x % order  # Make sure x is in [0, order), this is needed because x may be negative
        mul_inv[i] = x

    assert exp[order-1] == 1, "Alpha does not have multiplicative order 2^m - 1"
    assert len(set(exp[0:order-1])) == order - 1, "The anti-log LUT is not unique"
    assert len(set(log[1:order])) == order - 1, "The log LUT is not unique"
    assert len(set(mul_inv[1:order])) == order - 1, "The multiplicative inverse LUT is not unique; this should only happen is `p` is not prime"

    return exp, log, mul_inv


def GFp_factory(p, rebuild=False):
    """
    Factory function to construct Galois field array classes of type GF(p).

    Parameters
    ----------
    p : int
        The prime characteristic of the field GF(p).
    rebuild : bool, optional
        A flag to force a rebuild of the class and its lookup tables. Default is `False` which will return the cached,
        previously-built class if it exists.

    Returns
    -------
    galois.GFp
        A new Galois field class that is a sublcass of `galois.GFp`.
    """
    # pylint: disable=global-variable-undefined,protected-access
    assert is_prime(p)

    order = p
    name = "GF{}".format(order)

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = primitive_roots(p)[0]

    # If the requested field has already been constructed, return it instead of rebuilding
    if not rebuild and p in GFp_factory.classes:
        return GFp_factory.classes[p]

    assert 2 <= order < 2**16
    dtype = np.int64
    # if order < 2**8:
    #     dtype = np.uint8
    # else:
    #     dtype = np.uint16

    global MUL_INV, LOG, ORDER
    _, LOG, MUL_INV = _build_luts(p, alpha, dtype)
    ORDER = order

    # Create new class type
    cls = type(name, (GFp,), {
        "characteristic": p,
        "power": 1,
        "order": p,
        "alpha": alpha,
        "_dtype": dtype,
        "_MUL_INV": MUL_INV
    })

    cls.prim_poly = min_poly(cls.alpha, cls, 1)

    cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], nopython=True)(_ufunc_power)
    cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], nopython=True)(_ufunc_log)

    # Add class to dictionary of flyweights
    GFp_factory.classes[p] = cls

    return cls

GFp_factory.classes = {}
