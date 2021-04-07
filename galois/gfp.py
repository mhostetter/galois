import numba
import numpy as np

from .algorithm import gcd, gcd_jit
from .gf import GFArray

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
ORDER = None  # The field's order `p^m`
ALPHA = None  # The field's primitive element

# Placeholder functions to be replaced by JIT-compiled function
ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y
MULTIPLICATIVE_INVERSE_JIT = lambda x: 1 / x


class GFp(GFArray):
    """
    An abstract base class for all :math:`\\mathrm{GF}(p)` field array classes.
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
        if target not in ["cpu", "parallel", "cuda"]:
            raise ValueError(f"Argument `target` must be in ['cpu', 'parallel', 'cuda'], not {target}.")
        if mode not in ["lookup", "calculate"]:
            raise ValueError(f"Argument `mode` must be in ['lookup', 'calculate'], not {mode}.")
        if not isinstance(rebuild, bool):
            raise TypeError(f"Argument 'rebuild' must be a bool, not {type(rebuild)}.")

        global CHARACTERISTIC, ORDER, ALPHA, ADD_JIT, MULTIPLY_JIT, MULTIPLICATIVE_INVERSE_JIT

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        cls.ufunc_mode = mode
        cls.ufunc_target = target

        object_mode = cls.dtypes[-1] == np.object_

        if object_mode:
            cls._link_python_calculate_ufuncs()

        elif mode == "lookup":
            # Build the lookup tables if they don't exist or a rebuild is requested
            if cls._EXP is None or rebuild:
                cls._build_luts()

            # Compile ufuncs using standard EXP, LOG, and ZECH_LOG implementation
            cls._jit_compile_lookup_ufuncs(target)

        else:
            CHARACTERISTIC = cls.characteristic
            ORDER = cls.order
            ALPHA = int(cls.alpha)  # Convert from field element to integer

            # JIT-compile add and multiply routines for reference in polynomial evaluation routine
            ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(add_calculate)
            MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(multiply_calculate)
            MULTIPLICATIVE_INVERSE_JIT = numba.jit("int64(int64)", nopython=True)(multiplicative_inverse_calculate)

            # Create numba JIT-compiled ufuncs
            cls._ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(add_calculate)
            cls._ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(subtract_calculate)
            cls._ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(multiply_calculate)
            cls._ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(divide_calculate)
            cls._ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(additive_inverse_calculate)
            cls._ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(multiple_add_calculate)
            cls._ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(power_calculate)
            cls._ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(log_calculate)
            cls._ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(poly_eval_calculate)

    ###############################################################################
    # Galois field explicit arithmetic in pure python for extremely large fields
    ###############################################################################

    @classmethod
    def _add_calculate(cls, a, b):
        return (a + b) % cls.order

    @classmethod
    def _subtract_calculate(cls, a, b):
        return (a - b) % cls.order

    @classmethod
    def _multiply_calculate(cls, a, b):
        return (a * b) % cls.order

    @classmethod
    def _additive_inverse_calculate(cls, a):
        return (-a) % cls.order

    @classmethod
    def _multiplicative_inverse_calculate(cls, a):
        a_inv = gcd(a, cls.order)[1]
        return a_inv % cls.order


###############################################################################
# Galois field arithmetic, explicitly calculated wihtout lookup tables
###############################################################################

def add_calculate(a, b):  # pragma: no cover
    return (a + b) % ORDER


def subtract_calculate(a, b):  # pragma: no cover
    return (a - b) % ORDER


def multiply_calculate(a, b):  # pragma: no cover
    return (a * b) % ORDER


def divide_calculate(a, b):  # pragma: no cover
    if a == 0 or b == 0:
        # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0
    b_inv = MULTIPLICATIVE_INVERSE_JIT(b)
    return MULTIPLY_JIT(a, b_inv)


def additive_inverse_calculate(a):  # pragma: no cover
    return (-a) % ORDER


def multiplicative_inverse_calculate(a):  # pragma: no cover
    a_inv = gcd_jit(a, ORDER)[1]
    return a_inv % ORDER


def multiple_add_calculate(a, multiple):  # pragma: no cover
    return (a * multiple) % ORDER


def power_calculate(a, power):  # pragma: no cover
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


def log_calculate(beta):  # pragma: no cover
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


def poly_eval_calculate(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
