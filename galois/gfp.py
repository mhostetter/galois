import numba
import numpy as np

from .algorithm import extended_euclidean_algorithm
from .gf import GFBase, GFArray

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
ORDER = None  # The field's order `p^m`
ALPHA = None  # The field's primitive element

# Placeholder functions to be replaced by JIT-compiled function
ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y
MULTIPLICATIVE_INVERSE_JIT = lambda x: 1 / x


class GFp(GFBase, GFArray):
    """
    An abstract base class for all :math:`\\mathrm{GF}(p)` field array classes.

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
    galois.GFp
        The copied input array as a :math:`\\mathrm{GF}(p)` field array.

    Note
    ----
        This is an abstract base class for all :math:`\\mathrm{GF}(p)` fields. It cannot be instantiated directly.
        :math:`\\mathrm{GF}(p)` field classes are created using :obj:`galois.GF_factory`.
    """

    def __new__(cls, *args, **kwargs):
        if cls is GFp:
            raise NotImplementedError("GFp is an abstract base class that cannot be directly instantiated")
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def _build_luts(cls):

        dtype = np.int64
        if cls.order > np.iinfo(dtype).max:
            raise ValueError(f"Cannot build lookup tables for GF(p) class with order {cls.order} since the elements cannot be represented with dtype {dtype}")

        cls._EXP = np.zeros(2*cls.order, dtype=dtype)
        cls._LOG = np.zeros(cls.order, dtype=dtype)
        cls._ZECH_LOG = np.zeros(cls.order, dtype=dtype)

        cls._EXP[0] = 1
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, cls.order):
            # Increment the anti-log lookup table by multiplying by the primitive element alpha, which is
            # the "multiplicative generator"
            cls._EXP[i] = cls._EXP[i-1] * cls.alpha

            if cls._EXP[i] >= cls.order:
                cls._EXP[i] = cls._EXP[i] % cls.order

            # Assign to the log lookup but skip indices greater than or equal to `order-1`
            # because `EXP[0] == EXP[order-1]``
            if i < cls.order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, cls.order):
            a_i = cls._EXP[i]  # alpha^i
            cls._ZECH_LOG[i] = cls._LOG[(1 + a_i) % cls.order]  # Addition in GF(p)

        assert cls._EXP[cls.order-1] == 1, f"Primitive element `alpha = {cls.alpha}` does not have multiplicative order `order - 1 = {cls.order-1}` and therefore isn't a multiplicative generator for GF({cls.order})"
        assert len(set(cls._EXP[0:cls.order-1])) == cls.order - 1, "The anti-log lookup table is not unique"
        assert len(set(cls._LOG[1:cls.order])) == cls.order - 1, "The log lookup table is not unique"

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[cls.order:2*cls.order] = cls._EXP[1:1 + cls.order]

    @classmethod
    def target(cls, target, mode, rebuild=False):
        """
        Retarget the just-in-time compiled numba ufuncs.

        Parameters
        ----------
        target : str
            The `target` keyword argument from :obj:`numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`.
        mode : str
            The type of field computation, either `"lookup"` or `"calculate"`. The "lookup" mode will use Zech log, log,
            and anti-log lookup tables for speed. The "calculate" mode will not store any lookup tables, but perform field
            arithmetic on the fly. The "calculate" mode is designed for large fields that cannot store lookup tables in RAM.
            Generally, "calculate" will be slower than "lookup".
        rebuild : bool, optional
            Indicates whether to force a rebuild of the lookup tables. The default is `False`.
        """
        global CHARACTERISTIC, ORDER, ALPHA, ADD_JIT, MULTIPLY_JIT, MULTIPLICATIVE_INVERSE_JIT  # pylint: disable=global-statement
        CHARACTERISTIC = cls.characteristic
        ORDER = cls.order
        ALPHA = int(cls.alpha)  # Convert from field element to integer

        if target not in ["cpu", "parallel", "cuda"]:
            raise ValueError(f"Valid numba compilation targets are ['cpu', 'parallel', 'cuda'], not {target}")
        if mode not in ["lookup", "calculate"]:
            raise ValueError(f"Valid GF(p) field calculation modes are ['lookup', 'calculate'], not {mode}")
        if not isinstance(rebuild, bool):
            raise TypeError(f"The 'rebuild' argument must be a bool, not {type(rebuild)}")

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        cls.ufunc_mode = mode
        cls.ufunc_target = target

        if mode == "lookup":
            # Build the lookup tables if they don't exist or a rebuild is requested
            if cls._EXP is None or rebuild:
                cls._build_luts()

            # Compile ufuncs using standard EXP, LOG, and ZECH_LOG implementation
            cls._compile_lookup_ufuncs(target)
        else:
            # JIT-compile add and multiply routines for reference in polynomial evaluation routine
            ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(add_calculate)
            MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(multiply_calculate)
            MULTIPLICATIVE_INVERSE_JIT = numba.jit("int64(int64)", nopython=True)(multiplicative_inverse_calculate)

            # Create numba JIT-compiled ufuncs
            cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(add_calculate)
            cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(subtract_calculate)
            cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(multiply_calculate)
            cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(divide_calculate)
            cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(additive_inverse_calculate)
            cls._numba_ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(multiple_add_calculate)
            cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(power_calculate)
            cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(log_calculate)
            cls._numba_ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(poly_eval_calculate)


###############################################################################
# Galois field arithmetic, explicitly calculated wihtout lookup tables
###############################################################################

def add_calculate(a, b):
    return (a + b) % ORDER


def subtract_calculate(a, b):
    return (a - b) % ORDER


def multiply_calculate(a, b):
    return (a * b) % ORDER


def divide_calculate(a, b):
    if a == 0 or b == 0:
        # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0
    b_inv = MULTIPLICATIVE_INVERSE_JIT(b)
    return MULTIPLY_JIT(a, b_inv)


def additive_inverse_calculate(a):
    return (-a) % ORDER


def multiplicative_inverse_calculate(a):
    a_inv = extended_euclidean_algorithm(a, ORDER)[0]
    return a_inv % ORDER


def multiple_add_calculate(a, multiple):
    return (a * multiple) % ORDER


def power_calculate(a, power):
    """
    Square and Multiply Algorithm

    a^13 = (1) * (a)^13
         = (a) * (a)^12
         = (a) * (a^2)^6
         = (a) * (a^4)^3
         = (a * a^4) * (a^4)^2
         = (a * a^4) * (a^8)
         = result_m * result_s
    """
    # NOTE: The a == 0 and b < 0 condition will be caught outside of the the ufunc and raise ZeroDivisonError
    if power == 0:
        return 1
    elif power < 0:
        a = MULTIPLICATIVE_INVERSE_JIT(a)
        power = abs(power)

    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = MULTIPLY_JIT(result_s, result_s)
            power //= 2
        else:
            result_m = MULTIPLY_JIT(result_m, result_s)
            power -= 1

    result = MULTIPLY_JIT(result_m, result_s)

    return result


def log_calculate(beta):
    """
    TODO: Replace this with more efficient algorithm

    alpha in GF(p^m) and generates field
    beta in GF(p^m)

    gamma = log_alpha(beta), such that: alpha^gamma = beta
    """
    # Naive algorithm
    result = 1
    for i in range(0, ORDER-1):
        if result == beta:
            break
        result = MULTIPLY_JIT(result, ALPHA)
    return i


def poly_eval_calculate(coeffs, values, results):
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
