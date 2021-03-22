import numba

from .gf import GFBase, GFArray, DTYPES

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field

# Placeholder functions to be replaced by JIT-compiled function
ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y


class GF2(GFBase, GFArray):
    """
    Galois field array class for :math:`\\mathrm{GF}(2)` fields.

    Parameters
    ----------
    array : array_like
        The input array to be converted to a Galois field array. The input array is copied, so the original array
        is unmodified by changes to the Galois field array. Valid input array types are :obj:`numpy.ndarray`,
        :obj:`list`, :obj:`tuple`, or :obj:`int`.
    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is :obj:`numpy.int64`.

    Returns
    -------
    galois.GF2
        The copied input array as a :math:`\\mathrm{GF}(2)` field array.

    Examples
    --------

    GF2 class properties

    .. ipython:: python

        print(galois.GF2)
        galois.GF2.characteristic
        galois.GF2.degree
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
    degree = 1
    order = 2
    prim_poly = None  # Will set this in __init__.py
    alpha = 1
    dtypes = DTYPES

    @classmethod
    def target(cls, target):
        """
        Retarget the just-in-time compiled `numba` ufuncs.

        Parameters
        ----------
        target : str
            The `target` keyword argument from :obj:`numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`.
        """
        global CHARACTERISTIC, ADD_JIT, MULTIPLY_JIT  # pylint: disable=global-statement
        CHARACTERISTIC = cls.characteristic

        if target not in ["cpu", "parallel", "cuda"]:
            raise ValueError(f"Valid numba compilation targets are ['cpu', 'parallel', 'cuda'], not {target}")

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        cls.ufunc_mode = "calculate"
        cls.ufunc_target = target

        # JIT-compile add and multiply routines for reference in polynomial evaluation routine
        ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(_add_calculate)
        MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(_multiply_calculate)

        # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
        cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_calculate)
        cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_calculate)
        cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
        cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_calculate)
        cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(_negative_calculate)
        cls._numba_ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiple_add_calculate)
        cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
        cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(_log_calculate)
        cls._numba_ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_calculate)


###############################################################################
# Galois field arithmetic, explicitly calculated wihtout lookup tables
###############################################################################

def _add_calculate(a, b):
    return a ^ b


def _subtract_calculate(a, b):
    return a ^ b


def _multiply_calculate(a, b):
    return a & b


def _divide_calculate(a, b):
    if b == 0:
        # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0
    else:
        return a


def _negative_calculate(a):
    return a


def _multiple_add_calculate(a, multiple):
    multiple = multiple % CHARACTERISTIC
    return MULTIPLY_JIT(a, multiple)


def _power_calculate(a, power):
    # NOTE: The a == 0 and b < 0 condition will be caught outside of the the ufunc and raise ZeroDivisonError
    if power == 0:
        return 1
    elif a == 0:
        return 0
    else:
        return a


def _log_calculate(a):  # pylint: disable=unused-argument
    # NOTE: The a == 0 condition will be caught outside of the ufunc and raise ArithmeticError
    return 0


def _poly_eval_calculate(coeffs, values, results):
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
