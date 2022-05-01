"""
A module that defines a mixin classes for Galois field arithmetic.
"""
from __future__ import annotations

from typing import List

import numba
import numpy as np

from .._domains import _arithmetic
from .._domains._array import DTYPES
from .._domains._ufunc import FieldUfuncs


class FieldUfuncs_2_1(FieldUfuncs):
    """
    A mixin class that provides explicit calculation arithmetic for GF(2).
    """
    # Need to have a unique cache of "calculate" functions for GF(2)
    _FUNC_CACHE_CALCULATE = {}

    _UFUNC_OVERRIDES = {
        "add": np.bitwise_xor,
        "negative": np.positive,
        "subtract": np.bitwise_xor,
        "multiply": np.bitwise_and,
        "reciprocal": np.positive,
        "divide": np.bitwise_and,
    }

    ###############################################################################
    # Override ufunc routines to use native numpy bitwise ufuncs for GF(2)
    # arithmetic, which is faster than custom ufuncs
    ###############################################################################

    @classmethod
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

    @classmethod
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

    @classmethod
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
    def _reciprocal_calculate(a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        return 1

    @staticmethod
    def _divide_calculate(a: int, b) -> int:
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        return a & b

    @staticmethod
    @numba.extending.register_jitable
    def _power_calculate(a: int, b: int) -> int:
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1

        return a

    @staticmethod
    @numba.extending.register_jitable
    def _log_calculate(a: int, b: int) -> int:
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        if b != 1:
            raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")

        return 0

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a: FieldUfuncs_2_1) -> FieldUfuncs_2_1:
        return a.copy()


class FieldUfuncs_2_m(
    _arithmetic.MultiplyBinary,
    _arithmetic.ReciprocalFermat,
    _arithmetic.Divide,
    _arithmetic.FieldPowerSquareAndMultiply,
    _arithmetic.LogBruteForce,
    FieldUfuncs,
):
    """
    A mixin class that provides explicit calculation arithmetic for all GF(2^m) classes.
    """
    # Need to have a unique cache of "calculate" functions for GF(2^m)
    _FUNC_CACHE_CALCULATE = {}

    _UFUNC_OVERRIDES = {
        "add": np.bitwise_xor,
        "negative": np.positive,
        "subtract": np.bitwise_xor,
    }

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a: FieldUfuncs_2_m) -> FieldUfuncs_2_m:
        """
        Fact 3.42 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        field = type(a)
        return a ** (field.characteristic**(field.degree - 1))


class FieldUfuncs_p_1(
    _arithmetic.AddModular,
    _arithmetic.NegativeModular,
    _arithmetic.SubtractModular,
    _arithmetic.MultiplyModular,
    _arithmetic.ReciprocalModularEGCD,
    _arithmetic.Divide,
    _arithmetic.FieldPowerSquareAndMultiply,
    _arithmetic.LogBruteForce,
    _arithmetic.Sqrt,
    FieldUfuncs
):
    """
    A mixin class that provides explicit calculation arithmetic for all GF(p) classes.
    """
    # Need to have a unique cache of "calculate" functions for GF(p)
    _FUNC_CACHE_CALCULATE = {}

    @classmethod
    def _determine_dtypes(cls) -> List[np.dtype]:
        """
        The only valid dtypes are ones that can hold x*x for x in [0, order).
        """
        max_dtype = DTYPES[-1]
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

    @classmethod
    def _ufunc(cls, name: str):
        # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835548405.
        if cls.ufunc_mode == "jit-lookup" and name in ["add", "negative", "subtract"]:
            return cls._ufunc_calculate(name)
        else:
            return super()._ufunc(name)


class FieldUfuncs_p_m(
    _arithmetic.AddVector,
    _arithmetic.NegativeVector,
    _arithmetic.SubtractVector,
    _arithmetic.MultiplyVector,
    _arithmetic.ReciprocalFermat,
    _arithmetic.Divide,
    _arithmetic.FieldPowerSquareAndMultiply,
    _arithmetic.LogBruteForce,
    _arithmetic.Sqrt,
    FieldUfuncs
):
    """
    A mixin class that provides explicit calculation arithmetic for all GF(p^m) classes.
    """
    # Need to have a unique cache of "calculate" functions for GF(p^m)
    _FUNC_CACHE_CALCULATE = {}

    @classmethod
    def _determine_dtypes(cls) -> List[np.dtype]:
        """
        The only valid dtypes are ones that can hold x*x for x in [0, order).
        """
        # TODO: Is this correct?
        max_dtype = DTYPES[-1]
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    @classmethod
    def _convert_inputs_to_vector(cls, inputs, kwargs):
        v_inputs = list(inputs)
        for i in range(len(inputs)):
            if issubclass(type(inputs[i]), cls):
                v_inputs[i] = inputs[i].vector()

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if issubclass(type(output), cls):
                    o = output.vector()
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    @classmethod
    def _convert_output_from_vector(cls, output, field, dtype):  # pylint: disable=no-self-use
        if output is None:
            return None
        else:
            return field.Vector(output, dtype=dtype)

    @classmethod
    def _ufunc_routine_add(cls, ufunc, method, inputs, kwargs, meta):
        if cls.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super()._ufunc_routine_add(ufunc, method, inputs, kwargs, meta)
        else:
            # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert back to GF(p^m)
            cls._verify_operands_in_same_field(ufunc, inputs, meta)
            inputs, kwargs = cls._convert_inputs_to_vector(inputs, kwargs)
            output = getattr(ufunc, method)(*inputs, **kwargs)
            output = cls._convert_output_from_vector(output, meta["field"], meta["dtype"])
            return output

    @classmethod
    def _ufunc_routine_negative(cls, ufunc, method, inputs, kwargs, meta):
        if cls.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super()._ufunc_routine_negative(ufunc, method, inputs, kwargs, meta)
        else:
            # Convert entire array to polynomial/vector representation and perform array operation in GF(p)
            cls._verify_operands_in_same_field(ufunc, inputs, meta)
            inputs, kwargs = cls._convert_inputs_to_vector(inputs, kwargs)
            output = getattr(ufunc, method)(*inputs, **kwargs)
            output = cls._convert_output_from_vector(output, meta["field"], meta["dtype"])
            return output

    @classmethod
    def _ufunc_routine_subtract(cls, ufunc, method, inputs, kwargs, meta):
        if cls.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super()._ufunc_routine_subtract(ufunc, method, inputs, kwargs, meta)
        else:
            # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert back to GF(p^m)
            cls._verify_operands_in_same_field(ufunc, inputs, meta)
            inputs, kwargs = cls._convert_inputs_to_vector(inputs, kwargs)
            output = getattr(ufunc, method)(*inputs, **kwargs)
            output = cls._convert_output_from_vector(output, meta["field"], meta["dtype"])
            return output
