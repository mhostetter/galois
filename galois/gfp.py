import numpy as np

from .algorithm import extended_euclidean_algorithm, modular_exp
from .gf import GFBase

ORDER = None
EXP = []
LOG = []
MUL_INV = []


class GFpBase(GFBase):
    """
    asdf

    .. note::
        This is an abstract base class for all GF(p) fields. It cannot be instantiated directly.
    """

    def __new__(cls, *args, **kwargs):
        if cls is GFpBase:
            raise NotImplementedError("GFpBase is an abstract base class that should not be directly instantiated")
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
        np.ndarray
            The multiplicative inverse lookup table for the field. `MUL_INV[i] = 1/i`.
        """
        alpha = cls.alpha
        order = cls.order
        dtype = np.int64
        assert order <= np.iinfo(dtype).max

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

        cls._EXP = exp
        cls._LOG = log
        cls._MUL_INV = mul_inv

    @classmethod
    def _export_globals(cls):
        # Export lookup tables to global variables so JIT compiling can cache the tables in the binaries
        global ORDER, EXP, LOG, MUL_INV # pylint: disable=global-statement
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        MUL_INV = cls._MUL_INV

        super()._export_globals()

    @staticmethod
    def _add(a, b):
        # Calculate a + b
        return (a + b) % ORDER

    @staticmethod
    def _subtract(a, b):
        # Calculate a - b
        return (a - b) % ORDER

    @staticmethod
    def _multiply(a, b):
        # Calculate a * b
        return (a * b) % ORDER

    @staticmethod
    def _divide(a, b):
        # Calculate a / b
        return (a * MUL_INV[b]) % ORDER

    @staticmethod
    def _negative(a):
        # Calculate -a
        return (-a) % ORDER

    @staticmethod
    def _power(a, b):
        # Calculate a**b
        result = 1
        if b < 0:
            a = MUL_INV[a]
            b = abs(b)
        for _ in range(0, b):
            result = (result * a) % ORDER
        return result

    @staticmethod
    def _log(a):
        # Calculate np.log(a)
        return LOG[a]

    @staticmethod
    def _poly_eval(coeffs, values, results):
        # Calculate p(a)
        def _add(a, b):
            return (a + b) % ORDER

        def _multiply(a, b):
            return (a * b) % ORDER

        for i in range(values.size):
            results[i] = coeffs[0]
            for j in range(1, coeffs.size):
                results[i] = _add(coeffs[j], _multiply(results[i], values[i]))
