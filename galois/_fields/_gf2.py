import numba
import numpy as np

from .._overrides import set_module
from .._polys import Poly

from ._array import FieldArrayClass, DirMeta, FieldArray

__all__ = ["GF2"]


class GF2Meta(FieldArrayClass, DirMeta):
    """
    A metaclass for the GF(2) class.
    """
    # pylint: disable=no-value-for-parameter

    # Need to have a unique cache of "calculate" functions for GF(2)
    _FUNC_CACHE_CALCULATE = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._prime_subfield = cls
        cls._is_primitive_poly = True

        cls.compile(kwargs["compile"])

    @property
    def ufunc_modes(cls):
        return ["jit-calculate"]

    @property
    def default_ufunc_mode(cls):
        return "jit-calculate"

    def _compile_ufuncs(cls):
        super()._compile_ufuncs()
        assert cls.ufunc_mode == "jit-calculate"

        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["negative"] = np.positive
        cls._ufuncs["subtract"] = np.bitwise_xor
        cls._ufuncs["multiply"] = np.bitwise_and
        cls._ufuncs["reciprocal"] = np.positive
        cls._ufuncs["divide"] = np.bitwise_and

    ###############################################################################
    # Override ufunc routines to use native numpy bitwise ufuncs for GF(2)
    # arithmetic, which is faster than custom ufuncs
    ###############################################################################

    def _ufunc_routine_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, b in GF(2)
        b = 1 / a, a = 1 is the only valid element with a multiplicative inverse, which is 1
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        if np.count_nonzero(inputs[0]) != inputs[0].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        output = getattr(cls._ufunc("reciprocal"), method)(*inputs, **kwargs)
        return output

    def _ufunc_routine_divide(cls, ufunc, method, inputs, kwargs, meta):
        """
        Need to re-implement this to manually throw ZeroDivisionError if necessary
        """
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        if np.count_nonzero(inputs[meta["operands"][-1]]) != inputs[meta["operands"][-1]].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        output = getattr(cls._ufunc("divide"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, c in GF(2)
        c = a ** 2
          = a * a
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        return inputs[0]

    ###############################################################################
    # Arithmetic functions using explicit calculation
    ###############################################################################

    @staticmethod
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.positive()` is faster.
        """
        return a

    @staticmethod
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_and()` is faster.
        """
        return a & b

    @staticmethod
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        return 1

    @staticmethod
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        return a & b

    @staticmethod
    @numba.extending.register_jitable
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1

        return a

    @staticmethod
    @numba.extending.register_jitable
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        if b != 1:
            raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")

        return 0

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a):
        return a.copy()


@set_module("galois")
class GF2(FieldArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, compile="jit-calculate"):
    r"""
    A :obj:`~numpy.ndarray` subclass over :math:`\mathrm{GF}(2)`.

    Important
    ---------
        This class is a pre-generated :obj:`~galois.FieldArray` subclass generated with `galois.GF(2)` and is included in the API
        for convenience.

        Only the constructor is documented on this page. See :obj:`~galois.FieldArray` for all other classmethods and methods
        for :obj:`~galois.GF2`.

    See :ref:`Galois Field Classes` for a detailed discussion of the relationship between :obj:`~galois.FieldArrayClass` and
    :obj:`~galois.FieldArray`.

    See :ref:`Array Creation` for a detailed discussion on creating arrays (with and without copying) from array-like
    objects, valid NumPy data types, and other :obj:`~galois.FieldArray` classmethods.

    Examples
    --------
    This class is equivalent, and in fact identical, to the subclass returned from the class factory :func:`~galois.GF`.

    .. ipython:: python

        galois.GF2 is galois.GF(2)
        print(galois.GF2)

    The *Galois field array class* :obj:`~galois.GF2` is a subclass of :obj:`~galois.FieldArray`, with :obj:`~galois.FieldArrayClass` as its
    metaclass.

    .. ipython:: python

        isinstance(galois.GF2, galois.FieldArrayClass)
        issubclass(galois.GF2, galois.FieldArray)

    Create a :ref:`Galois field array` using :obj:`~galois.GF2`'s constructor.

    .. ipython:: python

        x = galois.GF2([1, 0, 1, 1]); x

    The *Galois field array* `x` is an instance of the *Galois field array class* :obj:`~galois.GF2`.

    .. ipython:: python

        isinstance(x, galois.GF2)
    """

# Define the GF(2) primitive polynomial here, not in _fields/_gf2.py, to avoid a circular dependency with `Poly`.
# The primitive polynomial is p(x) = x - alpha, where alpha = 1. Over GF(2), this is equivalent
# to p(x) = x + 1.
GF2._irreducible_poly = Poly([1, 1], field=GF2)  # pylint: disable=protected-access
