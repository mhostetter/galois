import numba
import numpy as np

from .._overrides import set_module

from ._array import FieldArray
from ._class import FieldClass, DirMeta
from ._ufunc import  _FUNCTION_TYPE

__all__ = ["GF2"]


class GF2Meta(FieldClass, DirMeta):
    """
    A metaclass for the GF(2) class.
    """
    # pylint: disable=abstract-method,no-value-for-parameter

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
    # Individual JIT arithmetic functions, pre-compiled (cached)
    ###############################################################################

    def _calculate_jit(cls, name):
        return compile_jit(name)

    def _python_func(cls, name):
        return eval(f"{name}")

    ###############################################################################
    # Individual ufuncs, compiled on-demand
    ###############################################################################

    def _calculate_ufunc(cls, name):
        return compile_ufunc(name, cls.characteristic, cls.degree, cls._irreducible_poly_int)

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
# Compile functions
###############################################################################

CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
DEGREE = None  # The prime power `m` of the Galois field
IRREDUCIBLE_POLY = None  # The field's primitive polynomial in integer form

# pylint: disable=redefined-outer-name,unused-argument


def compile_jit(name):
    """
    Compile a JIT arithmetic function. These can be cached.
    """
    if name not in compile_jit.cache:
        function = eval(f"{name}")
        if _FUNCTION_TYPE[name] == "unary":
            compile_jit.cache[name] = numba.jit(["int64(int64, int64, int64, int64)"], nopython=True, cache=True)(function)
        else:
            compile_jit.cache[name] = numba.jit(["int64(int64, int64, int64, int64, int64)"], nopython=True, cache=True)(function)
    return compile_jit.cache[name]

compile_jit.cache = {}


def compile_ufunc(name, CHARACTERISTIC_, DEGREE_, IRREDUCIBLE_POLY_):
    """
    Compile an arithmetic ufunc. These cannot be cached as the field parameters are compiled into the binary.
    """
    key = (name, CHARACTERISTIC_, DEGREE_, IRREDUCIBLE_POLY_)
    if key not in compile_ufunc.cache:
        global CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        CHARACTERISTIC = CHARACTERISTIC_
        DEGREE = DEGREE_
        IRREDUCIBLE_POLY = IRREDUCIBLE_POLY_

        function = eval(f"{name}_ufunc")
        if _FUNCTION_TYPE[name] == "unary":
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64)"], nopython=True)(function)
        else:
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(function)

    return compile_ufunc.cache[key]

compile_ufunc.cache = {}


###############################################################################
# Arithmetic explicitly calculated
###############################################################################

def add(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return a ^ b


def negative(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return a


def subtract(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return a ^ b


def multiply(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    return a & b


@numba.extending.register_jitable(inline="always")
def reciprocal(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if a == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    return 1


def reciprocal_ufunc(a):  # pragma: no cover
    return reciprocal(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def divide(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    return a & b


def divide_ufunc(a, b):  # pragma: no cover
    return divide(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def power(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if a == 0 and b < 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if b == 0:
        return 1
    else:
        return a


def power_ufunc(a, b):  # pragma: no cover
    return power(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def log(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if a == 0:
        raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
    if b != 1:
        raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")

    return 0


def log_ufunc(a, b):  # pragma: no cover
    return log(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


###############################################################################
# A pre-generated FieldArray subclass for GF(2)
###############################################################################

@set_module("galois")
class GF2(FieldArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, compile="jit-calculate"):
    r"""
    Creates an array over :math:`\mathrm{GF}(2)`.

    This class is a pre-generated :obj:`galois.FieldArray` subclass generated with `galois.GF(2)` and is included in the API
    for convenience. See :obj:`galois.FieldArray` and :obj:`galois.FieldClass` for more complete documentation and examples.

    Parameters
    ----------
    array : int, str, tuple, list, numpy.ndarray, galois.FieldArray
        The input array-like object to be converted to a Galois field array. See the examples section for demonstations of array creation
        using each input type. See see :func:`galois.FieldClass.display` and :obj:`galois.FieldClass.display_mode` for a description of the
        "integer" and "polynomial" representation of Galois field elements.

        * :obj:`int`: A single integer, which is the "integer representation" of a Galois field element, creates a 0-D array.
        * :obj:`str`: A single string, which is the "polynomial representation" of a Galois field element, creates a 0-D array.
        * :obj:`tuple`, :obj:`list`: A list or tuple (or nested lists/tuples) of ints or strings (which can be mix-and-matched) creates an array of
          Galois field elements from their integer or polynomial representations.
        * :obj:`numpy.ndarray`, :obj:`galois.FieldArray`: An array of ints creates a copy of the array over this specific field.

    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
        dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.
    copy : bool, optional
        The `copy` keyword argument from :func:`numpy.array`. The default is `True` which makes a copy of the input array.
    order : str, optional
        The `order` keyword argument from :func:`numpy.array`. Valid values are `"K"` (default), `"A"`, `"C"`, or `"F"`.
    ndmin : int, optional
        The `ndmin` keyword argument from :func:`numpy.array`. The minimum number of dimensions of the output.
        The default is 0.

    Returns
    -------
    galois.FieldArray
        The copied input array as a Galois field array over :math:`\mathrm{GF}(2)`.

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
