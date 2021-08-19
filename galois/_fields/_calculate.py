"""
A module that contains a metaclass mixin that provides generic Galois field arithmetic using explicit calculation.
"""
import numba
from numba import int64
import numpy as np

from ._properties import PropertiesMeta


class CalculateMeta(PropertiesMeta):
    """
    A mixin metaclass that provides Galois field arithmetic using explicit calculation.
    """
    # pylint: disable=no-value-for-parameter

    # Function signatures for JIT-compiled "calculate" arithmetic functions
    _UNARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64))
    _BINARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64, int64))

    _FUNC_CACHE_CALCULATE = {}
    _UFUNC_CACHE_CALCULATE = {}

    _UFUNC_TYPE = {
        "add": "binary",
        "negative": "unary",
        "subtract": "binary",
        "multiply": "binary",
        "reciprocal": "unary",
        "divide": "binary",
        "power": "binary",
        "log": "binary",
    }

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._reset_globals()

    def _set_globals(cls, name):  # pylint: disable=unused-argument,no-self-use
        """
        Sets the global variables in the GF*Meta metaclass mixins that are needed for linking during JIT compilation.
        """
        return

    def _reset_globals(cls):  # pylint: disable=no-self-use
        """
        Resets the global variables in the GF*Meta metaclass mixins so when the pure-python functions/ufuncs invoke these
        globals, they reference the correct pure-python functions and not the JIT-compiled functions/ufuncs.
        """
        return

    def _func_calculate(cls, name, reset=True):
        """
        Returns a JIT-compiled arithmetic function using explicit calculation. These functions are once-compiled and shared for all
        Galois fields. The only difference between Galois fields are the characteristic, degree, and irreducible polynomial that are
        passed in as inputs.
        """
        key = (name,)

        if key not in cls._FUNC_CACHE_CALCULATE:
            cls._set_globals(name)
            function = getattr(cls, f"_{name}_calculate")

            if cls._UFUNC_TYPE[name] == "unary":
                cls._FUNC_CACHE_CALCULATE[key] = numba.jit(["int64(int64, int64, int64, int64)"], nopython=True, cache=True)(function)
            else:
                cls._FUNC_CACHE_CALCULATE[key] = numba.jit(["int64(int64, int64, int64, int64, int64)"], nopython=True, cache=True)(function)

            if reset:
                cls._reset_globals()

        return cls._FUNC_CACHE_CALCULATE[key]

    def _ufunc_calculate(cls, name):
        """
        Returns a JIT-compiled arithmetic ufunc using explicit calculation. These ufuncs are compiled for each Galois field since
        the characteristic, degree, and irreducible polynomial are compiled into the ufuncs as constants.
        """
        key = (name, cls.characteristic, cls.degree, cls._irreducible_poly_int)

        if key not in cls._UFUNC_CACHE_CALCULATE:
            cls._set_globals(name)
            function = getattr(cls, f"_{name}_calculate")

            # These variables must be locals and not class properties for Numba to compile them as literals
            CHARACTERISTIC = cls.characteristic
            DEGREE = cls.degree
            IRREDUCIBLE_POLY = cls._irreducible_poly_int

            if cls._UFUNC_TYPE[name] == "unary":
                cls._UFUNC_CACHE_CALCULATE[key] = numba.vectorize(["int64(int64)"], nopython=True)(lambda a: function(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY))
            else:
                cls._UFUNC_CACHE_CALCULATE[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(lambda a, b: function(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY))

            cls._reset_globals()

        return cls._UFUNC_CACHE_CALCULATE[key]

    def _func_python(cls, name):
        """
        Returns a pure-python arithmetic function using explicit calculation. This lambda function wraps the arithmetic functions in
        GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta by passing in the field's characteristic, degree, and irreducible polynomial.
        """
        return getattr(cls, f"_{name}_calculate")
        # if cls._UFUNC_TYPE[name] == "unary":
        #     return lambda a: function(a, cls.characteristic, cls.degree, cls._irreducible_poly_int)
        # else:
        #     return lambda a, b: function(a, b, cls.characteristic, cls.degree, cls._irreducible_poly_int)

    def _ufunc_python(cls, name):
        """
        Returns a pure-python arithmetic ufunc using explicit calculation.
        """
        function = getattr(cls, f"_{name}_calculate")
        if cls._UFUNC_TYPE[name] == "unary":
            return np.frompyfunc(lambda a: function(a, cls.characteristic, cls.degree, cls._irreducible_poly_int), 1, 1)
        else:
            return np.frompyfunc(lambda a, b: function(a, b, cls.characteristic, cls.degree, cls._irreducible_poly_int), 2, 1)

    ###############################################################################
    # Arithmetic functions using explicit calculation
    ###############################################################################

    @staticmethod
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError
