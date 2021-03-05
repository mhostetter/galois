import numba
import numpy as np

from .algorithm import is_prime, extended_euclidean_algorithm, primitive_roots, modular_exp, min_poly
from .gf import _GF


class GFp(_GF):
    """
    asdf
    """

    _MUL_INV = []

    def __new__(cls, *args, **kwargs):
        assert cls is not GFp, "GFp is an abstract base class, it cannot be instantiated directly; use GFp_factory() to generate a GF(p) class"
        return super().__new__(cls, *args, **kwargs)


def _add(a, b):
    # Calculate a + b
    result = (a + b) % ORDER
    return result


def _subtract(a, b):
    # Calculate a - b
    result = (a - b) % ORDER
    return result


def _multiply(a, b):
    # Calculate a * b
    result = (a * b) % ORDER
    return result


def _divide(a, b):
    # Calculate a / b
    result = (a * MUL_INV[b]) % ORDER
    return result


def _negative(a):
    # Calculate -a
    result = (-a) % ORDER
    return result


def _power(a, b):
    # Calculate a**b
    result = 1
    if b < 0:
        a = MUL_INV[a]
        b = abs(b)
    for _ in range(0, b):
        result = (result * a) % ORDER
    return result


def _log(a):
    # Calculate np.log(a)
    result = LOG[a]
    return result


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

    # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
    cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], nopython=True)(_add)
    cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], nopython=True)(_subtract)
    cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], nopython=True)(_multiply)
    cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], nopython=True)(_divide)
    cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], nopython=True)(_negative)
    cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], nopython=True)(_power)
    cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], nopython=True)(_log)

    cls.prim_poly = min_poly(cls.alpha, cls, 1)

    # Add class to dictionary of flyweights
    GFp_factory.classes[p] = cls

    return cls

GFp_factory.classes = {}
