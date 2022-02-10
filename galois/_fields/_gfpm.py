"""
A module that contains a metaclass mixin that provides GF(p^m) arithmetic using explicit calculation.
"""
import numba
import numpy as np

from ._main import FieldClass, DirMeta
from ._dtypes import DTYPES

DTYPE = np.int64
INT_TO_POLY = lambda a, CHARACTERISTIC, DEGREE: [0,]*DEGREE
POLY_TO_INT = lambda a_vec, CHARACTERISTIC, DEGREE: 0
MULTIPLY = lambda a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY: a * b
RECIPROCAL = lambda a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY: 1 / a


class GFpmMeta(FieldClass, DirMeta):
    """
    A metaclass for all GF(p^m) classes.
    """
    # pylint: disable=no-value-for-parameter

    # Need to have a unique cache of "calculate" functions for GF(p^m)
    _FUNC_CACHE_CALCULATE = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly_coeffs = np.array(cls._irreducible_poly.coeffs, dtype=cls.dtypes[-1])
        cls._prime_subfield = kwargs["prime_subfield"]

        cls.compile(kwargs["compile"])

        # Determine if the irreducible polynomial is primitive
        if cls._is_primitive_poly is None:
            # TODO: Clean this up
            coeffs = cls.irreducible_poly.coeffs.view(np.ndarray).astype(cls.dtypes[-1])
            x = np.array(cls.primitive_element, dtype=cls.dtypes[-1], ndmin=1)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            cls._is_primitive_poly = cls._function_python("poly_evaluate")(coeffs, x, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)[0] == 0

    def _determine_dtypes(cls):
        """
        The only valid dtypes are ones that can hold x*x for x in [0, order).
        """
        # TODO: Is this correct?
        max_dtype = DTYPES[-1]
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

    def _set_globals(cls, name):
        super()._set_globals(name)
        global DTYPE, INT_TO_POLY, POLY_TO_INT, MULTIPLY, RECIPROCAL

        DTYPE = np.int64
        INT_TO_POLY = cls._func_calculate("int_to_poly", reset=False)
        POLY_TO_INT = cls._func_calculate("poly_to_int", reset=False)
        if name in ["reciprocal", "divide", "power", "log"]:
            MULTIPLY = cls._func_calculate("multiply", reset=False)
        if name in ["divide", "power"]:
            RECIPROCAL = cls._func_calculate("reciprocal", reset=False)

    def _reset_globals(cls):
        super()._reset_globals()
        global DTYPE, INT_TO_POLY, POLY_TO_INT, MULTIPLY, RECIPROCAL

        DTYPE = np.object_
        INT_TO_POLY = cls._int_to_poly
        POLY_TO_INT = cls._poly_to_int
        MULTIPLY = cls._func_python("multiply")
        RECIPROCAL = cls._func_python("reciprocal")

    def _func_calculate(cls, name, reset=True):
        key = (name,)

        if key not in cls._FUNC_CACHE_CALCULATE:
            # Generate extra JIT functions specific to the GF(p^m) field
            if name == "int_to_poly":
                cls._FUNC_CACHE_CALCULATE[key] = numba.jit(["int64[:](int64, int64, int64)"], nopython=True, cache=True)(cls._int_to_poly)
            elif name == "poly_to_int":
                cls._FUNC_CACHE_CALCULATE[key] = numba.jit(["int64(int64[:], int64, int64)"], nopython=True, cache=True)(cls._poly_to_int)
            else:
                super()._func_calculate(name, reset=reset)

        return cls._FUNC_CACHE_CALCULATE[key]

    ###############################################################################
    # Ufunc routines
    ###############################################################################

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

    def _convert_output_from_vector(cls, output, field, dtype):  # pylint: disable=no-self-use
        if output is None:
            return None
        else:
            return field.Vector(output, dtype=dtype)

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

    ###############################################################################
    # Arithmetic functions using explicit calculation
    #
    # NOTE: The ufunc inputs a and b are cast to integers at the beginning of each
    #       ufunc to prevent the non-JIT-compiled invocations (used in "large"
    #       fields with dtype=object) from performing infintely recursive
    #       arithmetic. Instead, the intended arithmetic inside the ufuncs is
    #       integer arithmetic.
    #       See https://github.com/mhostetter/galois/issues/253.
    ###############################################################################

    @staticmethod
    @numba.extending.register_jitable
    def _int_to_poly(a, CHARACTERISTIC, DEGREE):
        """
        Convert the integer representation to vector/polynomial representation
        """
        a = int(a)

        a_vec = np.zeros(DEGREE, dtype=DTYPE)
        for i in range(0, DEGREE):
            q = a // CHARACTERISTIC**(DEGREE - 1 - i)
            a -= q*CHARACTERISTIC**(DEGREE - 1 - i)
            a_vec[i] = q

        return a_vec

    @staticmethod
    @numba.extending.register_jitable
    def _poly_to_int(a_vec, CHARACTERISTIC, DEGREE):
        """
        Convert the integer representation to vector/polynomial representation
        """
        a = 0
        for i in range(0, DEGREE):
            a += a_vec[i]*CHARACTERISTIC**(DEGREE - 1 - i)

        return a

    @staticmethod
    @numba.extending.register_jitable
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        a = int(a)
        b = int(b)

        a_vec = INT_TO_POLY(a, CHARACTERISTIC, DEGREE)
        b_vec = INT_TO_POLY(b, CHARACTERISTIC, DEGREE)
        c_vec = (a_vec + b_vec) % CHARACTERISTIC
        c = POLY_TO_INT(c_vec, CHARACTERISTIC, DEGREE)

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        a = int(a)

        a_vec = INT_TO_POLY(a, CHARACTERISTIC, DEGREE)
        a_vec = (-a_vec) % CHARACTERISTIC
        c = POLY_TO_INT(a_vec, CHARACTERISTIC, DEGREE)

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        a = int(a)
        b = int(b)

        a_vec = INT_TO_POLY(a, CHARACTERISTIC, DEGREE)
        b_vec = INT_TO_POLY(b, CHARACTERISTIC, DEGREE)
        c_vec = (a_vec - b_vec) % CHARACTERISTIC
        c = POLY_TO_INT(c_vec, CHARACTERISTIC, DEGREE)

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        a = int(a)
        b = int(b)

        a_vec = INT_TO_POLY(a, CHARACTERISTIC, DEGREE)
        b_vec = INT_TO_POLY(b, CHARACTERISTIC, DEGREE)

        # The irreducible polynomial with the x^degree term removed
        irreducible_poly_vec = INT_TO_POLY(IRREDUCIBLE_POLY - CHARACTERISTIC**DEGREE, CHARACTERISTIC, DEGREE)

        c_vec = np.zeros(DEGREE, dtype=DTYPE)
        for _ in range(DEGREE):
            if b_vec[-1] > 0:
                c_vec = (c_vec + b_vec[-1]*a_vec) % CHARACTERISTIC

            # Multiply a(x) by x
            q = a_vec[0]
            a_vec[:-1] = a_vec[1:]
            a_vec[-1] = 0

            # Reduce a(x) modulo the irreducible polynomial
            if q > 0:
                a_vec = (a_vec - q*irreducible_poly_vec) % CHARACTERISTIC

            # Divide b(x) by x
            b_vec[1:] = b_vec[:-1]
            b_vec[0] = 0

        c = POLY_TO_INT(c_vec, CHARACTERISTIC, DEGREE)

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        From Fermat's Little Theorem:
        a^(p^m - 1) = 1 (mod p^m), for a in GF(p^m)

        a * a^-1 = 1
        a * a^-1 = a^(p^m - 1)
            a^-1 = a^(p^m - 2)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        ORDER = CHARACTERISTIC**DEGREE
        a = int(a)

        exponent = ORDER - 2
        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while exponent > 1:
            if exponent % 2 == 0:
                result_s = MULTIPLY(result_s, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                exponent //= 2
            else:
                result_m = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                exponent -= 1

        result = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

        return result

    @staticmethod
    @numba.extending.register_jitable
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        a = int(a)
        b = int(b)

        if a == 0:
            c = 0
        else:
            b_inv = RECIPROCAL(b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
            c = MULTIPLY(a, b_inv, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
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
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        a = int(a)
        b = int(b)

        if b == 0:
            return 1
        elif b < 0:
            a = RECIPROCAL(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
            b = abs(b)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while b > 1:
            if b % 2 == 0:
                result_s = MULTIPLY(result_s, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                b //= 2
            else:
                result_m = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
                b -= 1

        result = MULTIPLY(result_m, result_s, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

        return result

    @staticmethod
    @numba.extending.register_jitable
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        TODO: Replace this with more efficient algorithm
        a = α^m
        b is a primitive element of the field

        c = log(a, b)
        a = b^c
        """
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        ORDER = CHARACTERISTIC**DEGREE
        a = int(a)
        b = int(b)

        # Naive algorithm
        result = 1
        for i in range(0, ORDER - 1):
            if result == a:
                break
            result = MULTIPLY(result, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

        return i

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a):
        """
        Algorithm 3.34 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        Algorithm 3.36 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        field = type(a)
        p = field.characteristic

        if p % 4 == 3:
            roots = a ** ((field.order + 1)//4)

        elif p % 8 == 5:
            d = a ** ((field.order - 1)//4)
            roots = field.Zeros(a.shape)

            idxs = np.where(d == 1)
            roots[idxs] = a[idxs] ** ((field.order + 3)//8)

            idxs = np.where(d == p - 1)
            roots[idxs] = 2*a[idxs] * (4*a[idxs]) ** ((field.order - 5)//8)

        else:
            # Find a quadratic non-residue element `b`
            while True:
                b = field.Random(low=1)
                if not b.is_quadratic_residue():
                    break

            # Write p - 1 = 2^s * t
            n = field.order - 1
            s = 0
            while n % 2 == 0:
                n >>= 1
                s += 1
            t = n
            assert field.order - 1 == 2**s * t

            roots = field.Zeros(a.shape)  # Empty array of roots

            # Compute a root `r` for the non-zero elements
            idxs = np.where(a > 0)  # Indices where a has a reciprocal
            a_inv = np.reciprocal(a[idxs])
            c = b ** t
            r = a[idxs] ** ((t + 1)//2)
            for i in range(1, s):
                d = (r**2 * a_inv) ** (2**(s - i - 1))
                r[np.where(d == p - 1)] *= c
                c = c**2
            roots[idxs] = r  # Assign non-zero roots to the original array

        roots = np.minimum(roots, -roots).view(field)  # Return only the smaller root

        return roots
