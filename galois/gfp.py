import numba
import numpy as np

from .algorithm import extended_euclidean_algorithm, is_prime, primitive_root
from .gf import GFBase, DTYPES
from .poly import Poly

ORDER = None
EXP = []
LOG = []


def GFp_factory(p, mode="auto", rebuild=False):
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
    galois.GFpBase
        A new Galois field class that is a sublcass of `galois.GFpBase`.
    """
    if not isinstance(p, (int, np.integer)):
        raise TypeError(f"GF(p) prime characteristic `p` must be an integer, not {type(p)}")
    if not is_prime(p):
        return ValueError(f"GF(p) fields must have a prime characteristic `p`, not {p}")
    if not 2 <= p <= 2**16:
        return ValueError(f"GF(p) classes are only supported for 2 <= p <= 2**16, not {p}")

    # If the requested field has already been constructed, return it instead of rebuilding
    key = (p,)
    if not rebuild and key in GFp_factory.classes:
        return GFp_factory.classes[key]

    characteristic = p
    power = 1
    order = characteristic**power
    name = "GF{}".format(order)
    dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= order]

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = primitive_root(p)

    # Create new class type
    cls = type(name, (GFpBase,), {
        "characteristic": characteristic,
        "power": power,
        "order": order,
        "dtypes": dtypes
    })

    # Define the primitive element as a 0-dim array in the newly created Galois field array class
    cls.alpha = cls(alpha)

    # JIT compile the numba ufuncs
    cls.target("cpu", mode=mode, rebuild=rebuild)

    cls.prim_poly = Poly([1, -alpha], field=cls)  # pylint: disable=invalid-unary-operand-type

    # Add class to dictionary of flyweights
    GFp_factory.classes[key] = cls

    return cls

GFp_factory.classes = {}


class GFpBase(GFBase):
    """
    asdf

    .. note::
        This is an abstract base class for all GF(p) fields. It cannot be instantiated directly.
    """

    def __new__(cls, *args, **kwargs):
        if cls is GFpBase:
            raise NotImplementedError("GFpBase is an abstract base class that cannot be directly instantiated")
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def _build_luts(cls):
        """
        Constructs the multiplicative inverse lookup table.

        Parameters
        ----------
        dtype : np.dtype
            Numpy data type for lookup tables.

        Returns
        -------
        np.ndarray
            The anti-log lookup table for the field. `EXP[i] = alpha^i`.
        np.ndarray
            The log lookup table for the field. `LOG[i] = log_alpha(i)`.
        """
        dtype = np.int64
        if cls.order > np.iinfo(dtype).max:
            raise ValueError(f"Cannot build lookup tables for GF(p) class with order {cls.order} since the elements cannot be represented with dtype {dtype}")

        cls._EXP = np.zeros(2*cls.order, dtype=dtype)
        cls._LOG = np.zeros(cls.order, dtype=dtype)

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

        assert cls._EXP[cls.order-1] == 1, f"Primitive element `alpha = {cls.alpha}` does not have multiplicative order `order - 1 = {cls.order-1}` and therefore isn't a multiplicative generator for GF({cls.order})"
        assert len(set(cls._EXP[0:cls.order-1])) == cls.order - 1, "The anti-log lookup table is not unique"
        assert len(set(cls._LOG[1:cls.order])) == cls.order - 1, "The log lookup table is not unique"

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[cls.order:2*cls.order] = cls._EXP[1:1 + cls.order]

    @classmethod
    def target(cls, target, mode="lookup", rebuild=False):  # pylint: disable=arguments-differ
        """
        Retarget the just-in-time compiled numba ufuncs.

        Parameters
        ----------
        target : str
            The numba JIT `target` processor, either "cpu", "parallel", or "cuda".
        """
        if target not in ["cpu", "parallel", "cuda"]:
            raise ValueError(f"Valid numba compilation targets are ['cpu', 'parallel', 'cuda'], not {target}")
        if mode not in ["auto", "lookup", "calculate"]:
            raise ValueError(f"Valid GF(p) field calculation modes are ['auto', 'lookup' or 'calculate'], not {mode}")
        if not isinstance(rebuild, bool):
            raise ValueError(f"The 'rebuild' must be a bool, not {type(rebuild)}")

        if mode == "auto":
            mode = "lookup" if cls.order <= 2**16 else "calculate"

        global ORDER, EXP, LOG  # pylint: disable=global-statement
        ORDER = cls.order
        EXP = None
        LOG = None

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        if mode == "lookup":
            if cls._EXP is None or rebuild:
                cls._build_luts()

            # Export lookup tables to global variables so JIT compiling can cache the tables in the binaries
            EXP = cls._EXP
            LOG = cls._LOG

            # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
            cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_lookup)
            cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_lookup)
            cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_lookup)
            cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_lookup)
            cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(_negative_lookup)
            cls._numba_ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_lookup)
            cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_lookup)
            cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(_log_lookup)
        else:
            # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
            cls._numba_ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_calculate)
            cls._numba_ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_calculate)
            cls._numba_ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
            cls._numba_ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_calculate)
            cls._numba_ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(_negative_calculate)
            cls._numba_ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
            cls._numba_ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
            cls._numba_ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(_log_calculate)

        cls._numba_ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval)


###############################################################################
# Arithmetic functions using explicit calculation
###############################################################################

def _add_calculate(a, b):
    return (a + b) % ORDER


def _subtract_calculate(a, b):
    return (a - b) % ORDER


def _multiply_calculate(a, b):
    return (a * b) % ORDER


def _divide_calculate(a, b):
    if a == 0 or b == 0:
        # NOTE: The b == 0 condition will be caught outside of ufunc and raise ZeroDivisonError
        return 0
    b_inv = extended_euclidean_algorithm(b, ORDER)[0]
    return (a * b_inv) % ORDER
    # return (a * EXP[LOG[b] - 1]) % ORDER
    # return (a * MUL_INV[b]) % ORDER


def _negative_calculate(a):
    return (-a) % ORDER


def _power_calculate(a, b):
    if b < 0:
        a = extended_euclidean_algorithm(a, ORDER)[0]
        b = abs(b)
    result = 1
    for _ in range(0, b):
        result = (result * a) % ORDER
    return result


def _log_calculate(a):  # pylint: disable=unused-argument
    # return LOG[a]
    return 0


###############################################################################
# Arithmetic functions using lookup tables
###############################################################################

def _add_lookup(a, b):
    return (a + b) % ORDER


def _subtract_lookup(a, b):
    return (a - b) % ORDER


def _multiply_lookup(a, b):
    a = a % ORDER
    b = b % ORDER
    if a == 0 or b == 0:
        return 0
    else:
        # NOTE: We don't need `(LOG[a] + LOG[b]) % (ORDER - 1)` because we intentionally oversized the
        # anti-log table to avoid the modulo operation
        return EXP[LOG[a] + LOG[b]]


def _divide_lookup(a, b):
    if a == 0 or b == 0:
        # NOTE: The b == 0 condition will be caught outside of ufunc and raise ZeroDivisonError
        return 0
    else:
        return EXP[LOG[a] + (ORDER - 1) - LOG[b]]


def _negative_lookup(a):
    result = -a
    if result < 0:
        return ORDER + result
    return result


def _power_lookup(a, b):
    if b == 0:
        return 1
    elif a == 0:
        return 0
    else:
        return EXP[(LOG[a]*b) % (ORDER-1)]


def _log_lookup(a):
    return LOG[a]


def _poly_eval(coeffs, values, results):
    def _add(a, b):
        return (a + b) % ORDER

    def _multiply(a, b):
        return (a * b) % ORDER


    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = _add(coeffs[j], _multiply(results[i], values[i]))
