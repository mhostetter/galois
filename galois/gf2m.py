import numba
import numpy as np

from .gf import GFBase, GFArray

# Globals that will be set in target() and referenced in numba JIT-compiled functions
CHARACTERISTIC = None
ORDER = None

# FIXME
EXP = []
LOG = []


class GF2m(GFBase, GFArray):
    """
    An abstract base class for all :math:`\\mathrm{GF}(2^m)` field array classes.

    .. note::
        This is an abstract base class for all :math:`\\mathrm{GF}(2^m)` fields. It cannot be instantiated directly.

        :math:`\\mathrm{GF}(2^m)` field classes are created using `galois.GF_factory(2, m)` or `galois.GF2m_factory(m)`.

    Parameters
    ----------
    array : array_like
        The input array to be converted to a Galois field array. The input array is copied, so the original array
        is unmodified by the Galois field array. Valid input array types are `np.ndarray`, `list`, `tuple`, or `int`.
    dtype : np.dtype, optional
        The numpy `dtype` of the array elements. The default is `np.int64`. See: https://numpy.org/doc/stable/user/basics.types.html.
    """

    def __new__(cls, *args, **kwargs):
        if cls is GF2m:
            raise NotImplementedError("GF2m is an abstract base class that cannot be directly instantiated")
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def _build_luts(cls):
        prim_poly_dec = cls.prim_poly.decimal
        dtype = np.int64
        if cls.order > np.iinfo(dtype).max:
            raise ValueError(f"Cannot build lookup tables for GF(2^m) class with order {cls.order} since the elements cannot be represented with dtype {dtype}")

        cls._EXP = np.zeros(2*cls.order, dtype=dtype)
        cls._LOG = np.zeros(cls.order, dtype=dtype)
        cls._ZECH_LOG = np.zeros(cls.order, dtype=dtype)

        cls._EXP[0] = 1
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, cls.order):
            # Increment the anti-log lookup table by multiplying by the primitive element alpha, which is
            # the "multiplicative generator"
            cls._EXP[i] = cls._EXP[i - 1] * cls.alpha

            if cls._EXP[i] >= cls.order:
                cls._EXP[i] = cls._EXP[i] ^ prim_poly_dec

            assert 0 <= cls._EXP[i] < cls.order, "It is likely the polynomial {} is not primitive, given alpha^{} = {} is not contained within the field [0, {})".format(cls.prim_poly, i, cls._EXP[i], cls.order)

            # Assign to the log lookup but skip indices greater than or equal to `order-1`
            # because `EXP[0] == EXP[order-1]``
            if i < cls.order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, cls.order):
            a_i = cls._EXP[i]  # alpha^i
            cls._ZECH_LOG[i] = cls._LOG[1 ^ a_i]  # Addition in GF(2^m)

        assert cls._EXP[cls.order-1] == 1, f"Primitive element `alpha = {cls.alpha}` does not have multiplicative order `order - 1 = {cls.order-1}` and therefore isn't a multiplicative generator for GF({cls.order})"
        assert len(set(cls._EXP[0:cls.order-1])) == cls.order - 1, "The anti-log lookup table is not unique"
        assert len(set(cls._LOG[1:cls.order])) == cls.order - 1, "The log lookup table is not unique"

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[cls.order:2*cls.order] = cls._EXP[1:1 + cls.order]

    @classmethod
    def target(cls, target, mode="auto", rebuild=False):
        """
        Retarget the just-in-time compiled numba ufuncs.

        Parameters
        ----------
        target : str, optional
            The `target` from `numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. See: https://numba.readthedocs.io/en/stable/user/vectorize.html.
        mode : str, optional
            The type of field computation, either `"auto"`, `"lookup"`, or `"calculate"`. The default is `"auto"`.
            The "lookup" mode will use Zech log, log, and anti-log lookup table for speed. The `"calculate"` mode will
            not store any lookup tables, but calculate the field arithmetic on the fly. The `"calculate"` mode is slower
            than `"lookup"` but uses less RAM. The "auto" mode will determine whether to use `"lookup"` or `"calculate"` based
            on field order.
        rebuild : bool, optional
            Indicates whether to force a rebuild of the lookup tables. The default is `False`.
        """
        global CHARACTERISTIC, ORDER, EXP, LOG  # pylint: disable=global-statement
        CHARACTERISTIC = cls.characteristic
        ORDER = cls.order

        if target not in ["cpu", "parallel", "cuda"]:
            raise ValueError(f"Valid numba compilation targets are ['cpu', 'parallel', 'cuda'], not {target}")
        if mode not in ["auto", "lookup", "calculate"]:
            raise ValueError(f"Valid GF(2^m) field calculation modes are ['auto', 'lookup' or 'calculate'], not {mode}")
        if not isinstance(rebuild, bool):
            raise ValueError(f"The 'rebuild' must be a bool, not {type(rebuild)}")

        if mode == "auto":
            mode = "lookup" if cls.order <= 2**16 else "calculate"

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        assert mode == "lookup"  # FIXME
        if mode == "lookup":
            # Build the lookup tables if they don't exist or a rebuild is requested
            if cls._EXP is None or rebuild:
                cls._build_luts()

            # Compile ufuncs using standard EXP, LOG, and ZECH_LOG implementation
            cls._compile_lookup_ufuncs(target)

            # Overwrite some ufuncs for a more efficient implementation with characteristic 2
            cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(add_calculate)
            cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(subtract_calculate)
            cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(negative_calculate)
        else:
            pass
            # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
            # cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(add_calculate)
            # cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(subtract_calculate)
            # cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(multiply_calculate)
            # cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(divide_calculate)
            # cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(negative_calculate)
            # cls._numba_ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(multiple_calculate)
            # cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(power_calculate)
            # cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(log_calculate)

        # FIXME
        EXP = cls._EXP
        LOG = cls._LOG
        cls._numba_ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(poly_eval_calculate)


###############################################################################
# Galois field arithmetic using explicit calculation
###############################################################################

def add_calculate(a, b):
    return a ^ b


def subtract_calculate(a, b):
    return a ^ b


def negative_calculate(a):
    return a


def poly_eval_calculate(coeffs, values, results):
    def _add(a, b):
        return a ^ b

    def _multiply(a, b):
        if a == 0 or b == 0:
            return 0
        else:
            return EXP[LOG[a] + LOG[b]]

    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = _add(coeffs[j], _multiply(results[i], values[i]))
