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

    _dtype = np.int64


def _add(a, b):
    # Calculate a + b
    result = a ^ b
    return result


def _subtract(a, b):
    # Calculate a - b
    result = a ^ b
    return result


def _multiply(a, b):
    # Calculate a * b
    result = a & b
    return result


def _divide(a, b):
    # Calculate a / b
    result = a & b
    return result


def _negative(a):
    # Calculate -a
    result = a
    return result


def _power(a, b):
    # Calculate a**b
    if b == 0:
        result = 1
    elif a == 0:
        result = 0
    else:
        result = a
    return result


def _log(a):  # pylint: disable=unused-argument
    # Calculate np.log(a)
    result = 0
    return result


# Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
# pylint: disable=protected-access
GF2._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], nopython=True)(_add)
GF2._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], nopython=True)(_subtract)
GF2._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], nopython=True)(_multiply)
GF2._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], nopython=True)(_divide)
GF2._numba_ufunc_negative = numba.vectorize(["int64(int64)"], nopython=True)(_negative)
GF2._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], nopython=True)(_power)
GF2._numba_ufunc_log = numba.vectorize(["int64(int64)"], nopython=True)(_log)
