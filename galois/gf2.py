import numba

from .gf import GFBase, DTYPES

CHARACTERISTIC = None


class GF2(GFBase):
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
    prim_poly = None  # Will set this in __init__.py
    alpha = 1
    dtypes = DTYPES

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
            raise ValueError(f"Valid numba compilation targets are ['cpu', 'parallel', 'cuda'], not {target}")

        global CHARACTERISTIC  # pylint: disable=global-statement
        CHARACTERISTIC = cls.characteristic

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
        cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add)
        cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract)
        cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply)
        cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide)
        cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(_negative)
        cls._numba_ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiple_add)
        cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power)
        cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(_log)
        cls._numba_ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval)


def _add(a, b):
    return a ^ b


def _subtract(a, b):
    return a ^ b


def _multiply(a, b):
    return a & b


def _divide(a, b):
    return a & b


def _negative(a):
    return a


def _multiple_add(a, b):
    b = b % CHARACTERISTIC
    return a & b


def _power(a, b):
    if b == 0:
        return 1
    elif a == 0:
        return 0
    else:
        return a


def _log(a):  # pylint: disable=unused-argument
    return 0


def _poly_eval(coeffs, values, results):
    def _add(a, b):
        return a ^ b

    def _multiply(a, b):
        return a & b

    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = _add(coeffs[j], _multiply(results[i], values[i]))
