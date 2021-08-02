import numba
import numpy as np

from ._meta_class import FieldClass, DirMeta
from ._meta_ufunc import  _FUNCTION_TYPE


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
