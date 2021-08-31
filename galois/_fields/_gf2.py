"""
A module that contains a metaclass mixin that provides GF(2) arithmetic using explicit calculation.
"""
import numba
import numpy as np

from .._overrides import set_module

from ._array import FieldArray
from ._calculate import CalculateMeta
from ._class import FieldClass, DirMeta

__all__ = ["GF2"]


class GF2Meta(FieldClass, DirMeta, CalculateMeta):
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
    @numba.extending.register_jitable(inline="always")
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        else:
            return a

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        if b != 1:
            raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")

        return 0


@set_module("galois")
class GF2(FieldArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, compile="jit-calculate"):
    r"""
    An array over :math:`\mathrm{GF}(2)`.

    This class is a pre-generated :obj:`galois.FieldArray` subclass generated with `galois.GF(2)` and is included in the API
    for convenience. See :obj:`galois.FieldArray` and :obj:`galois.FieldClass` for more complete documentation and examples.

    Examples
    --------
    This class is equivalent (and, in fact, identical) to the class returned from the Galois field class constructor.

    .. ipython:: python

        print(galois.GF2)
        GF2 = galois.GF(2); print(GF2)
        GF2 is galois.GF2

    The Galois field properties can be viewed by class attributes, see :obj:`galois.FieldClass`.

    .. ipython:: python

        # View a summary of the field's properties
        print(galois.GF2.properties)

        # Or access each attribute individually
        galois.GF2.irreducible_poly
        galois.GF2.is_prime_field

    The class's constructor mimics the call signature of :func:`numpy.array`.

    .. ipython:: python

        # Construct a Galois field array from an iterable
        galois.GF2([1,0,1,1,0,0,0,1])

        # Or an iterable of iterables
        galois.GF2([[1,0], [1,1]])

        # Or a single integer
        galois.GF2(1)
    """
