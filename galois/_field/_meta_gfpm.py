import numba
import numpy as np

from ._dtypes import DTYPES
from ._meta_class import FieldClass, DirMeta
from ._meta_ufunc import  _FUNCTION_TYPE


class GFpmMeta(FieldClass, DirMeta):
    """
    A metaclass for all GF(p^m) classes.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly_coeffs = np.array(cls._irreducible_poly.coeffs, dtype=cls.dtypes[-1])
        cls._prime_subfield = kwargs["prime_subfield"]

        cls.compile(kwargs["compile"])

        # Determine if the irreducible polynomial is primitive
        if cls._is_primitive_poly is None:
            cls._is_primitive_poly = cls._poly_evaluate_python(cls._irreducible_poly.coeffs, cls.primitive_element) == 0

    @property
    def dtypes(cls):
        max_dtype = DTYPES[-1]
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(d) == 0:
            d = [np.object_]
        return d

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
    # Pure python arithmetic methods
    ###############################################################################

    def _int_to_poly(cls, a):
        a_vec = np.zeros(cls.degree, dtype=cls.dtypes[-1])
        for i in range(0, cls.degree):
            q = a // cls.characteristic**(cls.degree - 1 - i)
            a -= q*cls.characteristic**(cls.degree - 1 - i)
            a_vec[i] = q
        return a_vec

    def _poly_to_int(cls, a_vec):
        a = 0
        for i in range(0, cls.degree):
            a += a_vec[i]*cls.characteristic**(cls.degree - 1 - i)
        return a

    def _add_python(cls, a, b):
        a_vec = cls._int_to_poly(a)
        b_vec = cls._int_to_poly(b)
        c_vec = (a_vec + b_vec) % cls.characteristic
        return cls._poly_to_int(c_vec)

    def _negative_python(cls, a):
        a_vec = cls._int_to_poly(a)
        c_vec = (-a_vec) % cls.characteristic
        return cls._poly_to_int(c_vec)

    def _subtract_python(cls, a, b):
        a_vec = cls._int_to_poly(a)
        b_vec = cls._int_to_poly(b)
        c_vec = (a_vec - b_vec) % cls.characteristic
        return cls._poly_to_int(c_vec)

    def _multiply_python(cls, a, b):
        a_vec = cls._int_to_poly(a)
        b_vec = cls._int_to_poly(b)

        c_vec = np.zeros(cls.degree, dtype=cls.dtypes[-1])
        for _ in range(cls.degree):
            if b_vec[-1] > 0:
                c_vec = (c_vec + b_vec[-1]*a_vec) % cls.characteristic

            # Multiply a(x) by x
            q = a_vec[0]  # Don't need to divide by the leading coefficient of p(x) because it must be 1
            a_vec[:-1] = a_vec[1:]
            a_vec[-1] = 0

            # Reduce a(x) modulo the irreducible polynomial p(x)
            if q > 0:
                a_vec = (a_vec - q*cls._irreducible_poly_coeffs[1:]) % cls.characteristic

            # Divide b(x) by x
            b_vec[1:] = b_vec[:-1]
            b_vec[0] = 0

        return cls._poly_to_int(c_vec)

    def _reciprocal_python(cls, a):
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        exponent = cls.order - 2
        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while exponent > 1:
            if exponent % 2 == 0:
                result_s = cls._multiply_python(result_s, result_s)
                exponent //= 2
            else:
                result_m = cls._multiply_python(result_m, result_s)
                exponent -= 1

        result = cls._multiply_python(result_m, result_s)

        return result


###############################################################################
# Compile functions
###############################################################################

CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
DEGREE = None  # The prime power `m` of the Galois field
IRREDUCIBLE_POLY = None  # The field's primitive polynomial in integer form

DTYPE = np.int64
INT_TO_POLY = lambda a, CHARACTERISTIC, DEGREE: [0,]*DEGREE
POLY_TO_INT = lambda a_vec, CHARACTERISTIC, DEGREE: 0
MULTIPLY = lambda a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY: a * b
RECIPROCAL = lambda a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY: 1 / a

# pylint: disable=redefined-outer-name,unused-argument


def compile_jit(name, reset=True):
    """
    Compile a JIT arithmetic function. These can be cached.
    """
    if name not in compile_jit.cache:
        function = eval(f"{name}")

        if name == "int_to_poly":
            compile_jit.cache[name] = numba.jit(["int64[:](int64, int64, int64)"], nopython=True, cache=True)(function)
        elif name == "poly_to_int":
            compile_jit.cache[name] = numba.jit(["int64(int64[:], int64, int64)"], nopython=True, cache=True)(function)
        else:
            global DTYPE, INT_TO_POLY, POLY_TO_INT, MULTIPLY, RECIPROCAL

            DTYPE = np.int64
            INT_TO_POLY = compile_jit("int_to_poly", reset=False)
            POLY_TO_INT = compile_jit("poly_to_int", reset=False)

            if name in ["reciprocal", "divide", "power", "log"]:
                MULTIPLY = compile_jit("multiply", reset=False)
            if name in ["divide", "power"]:
                RECIPROCAL = compile_jit("reciprocal", reset=False)

            if _FUNCTION_TYPE[name] == "unary":
                compile_jit.cache[name] = numba.jit(["int64(int64, int64, int64, int64)"], nopython=True, cache=True)(function)
            else:
                compile_jit.cache[name] = numba.jit(["int64(int64, int64, int64, int64, int64)"], nopython=True, cache=True)(function)

            if reset:
                reset_globals()

    return compile_jit.cache[name]

compile_jit.cache = {}


def compile_ufunc(name, CHARACTERISTIC_, DEGREE_, IRREDUCIBLE_POLY_):
    """
    Compile an arithmetic ufunc. These cannot be cached as the field parameters are compiled into the binary.
    """
    key = (name, CHARACTERISTIC_, DEGREE_, IRREDUCIBLE_POLY_)
    if key not in compile_ufunc.cache:
        global CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY, DTYPE, INT_TO_POLY, POLY_TO_INT, MULTIPLY, RECIPROCAL
        CHARACTERISTIC = CHARACTERISTIC_
        DEGREE = DEGREE_
        IRREDUCIBLE_POLY = IRREDUCIBLE_POLY_

        DTYPE = np.int64
        INT_TO_POLY = compile_jit("int_to_poly", reset=False)
        POLY_TO_INT = compile_jit("poly_to_int", reset=False)

        if name in ["reciprocal", "divide", "power", "log"]:
            MULTIPLY = compile_jit("multiply", reset=False)
        if name in ["divide", "power"]:
            RECIPROCAL = compile_jit("reciprocal", reset=False)

        function = eval(f"{name}_ufunc")
        if _FUNCTION_TYPE[name] == "unary":
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64)"], nopython=True)(function)
        else:
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(function)

        reset_globals()

    return compile_ufunc.cache[key]

compile_ufunc.cache = {}


###############################################################################
# Helper functions
###############################################################################

def int_to_poly(a, CHARACTERISTIC, DEGREE):
    """
    Convert the integer representation to vector/polynomial representation
    """
    a_vec = np.zeros(DEGREE, dtype=DTYPE)
    for i in range(0, DEGREE):
        q = a // CHARACTERISTIC**(DEGREE - 1 - i)
        a -= q*CHARACTERISTIC**(DEGREE - 1 - i)
        a_vec[i] = q
    return a_vec


def poly_to_int(a_vec, CHARACTERISTIC, DEGREE):
    """
    Convert the integer representation to vector/polynomial representation
    """
    a = 0
    for i in range(0, DEGREE):
        a += a_vec[i]*CHARACTERISTIC**(DEGREE - 1 - i)
    return a


###############################################################################
# Arithmetic explicitly calculated
###############################################################################

@numba.extending.register_jitable(inline="always")
def add(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    a_vec = INT_TO_POLY(a, CHARACTERISTIC, DEGREE)
    b_vec = INT_TO_POLY(b, CHARACTERISTIC, DEGREE)
    c_vec = (a_vec + b_vec) % CHARACTERISTIC
    return POLY_TO_INT(c_vec, CHARACTERISTIC, DEGREE)


def add_ufunc(a, b):  # pragma: no cover
    return add(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def negative(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    a_vec = INT_TO_POLY(a, CHARACTERISTIC, DEGREE)
    a_vec = (-a_vec) % CHARACTERISTIC
    return POLY_TO_INT(a_vec, CHARACTERISTIC, DEGREE)


def negative_ufunc(a):  # pragma: no cover
    return negative(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def subtract(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    a_vec = INT_TO_POLY(a, CHARACTERISTIC, DEGREE)
    b_vec = INT_TO_POLY(b, CHARACTERISTIC, DEGREE)
    c_vec = (a_vec - b_vec) % CHARACTERISTIC
    return POLY_TO_INT(c_vec, CHARACTERISTIC, DEGREE)


def subtract_ufunc(a, b):  # pragma: no cover
    return subtract(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def multiply(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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

    return POLY_TO_INT(c_vec, CHARACTERISTIC, DEGREE)


def multiply_ufunc(a, b):  # pragma: no cover
    return multiply(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def reciprocal(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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


def reciprocal_ufunc(a):  # pragma: no cover
    return reciprocal(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def divide(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:
        return 0
    else:
        b_inv = RECIPROCAL(b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
        return MULTIPLY(a, b_inv, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


def divide_ufunc(a, b):  # pragma: no cover
    return divide(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def power(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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


def power_ufunc(a, b):  # pragma: no cover
    return power(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


@numba.extending.register_jitable(inline="always")
def log(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    """
    TODO: Replace this with more efficient algorithm

    a in GF(p^m)
    b in GF(p^m) and generates field

    c = log(a, b), such that b^a = c
    """
    if a == 0:
        raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

    # Naive algorithm
    ORDER = CHARACTERISTIC**DEGREE
    result = 1
    for i in range(0, ORDER - 1):
        if result == a:
            break
        result = MULTIPLY(result, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)

    return i


def log_ufunc(a, b):  # pragma: no cover
    return log(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)


def reset_globals():
    """
    Reset the global variable so when the pure-python ufuncs call these routines, they reference
    the correct pure-python functions (not JIT functions or JIT-compiled ufuncs).
    """
    global DTYPE, INT_TO_POLY, POLY_TO_INT, MULTIPLY, RECIPROCAL
    DTYPE = np.object_
    INT_TO_POLY = int_to_poly
    POLY_TO_INT = poly_to_int
    MULTIPLY = multiply
    RECIPROCAL = reciprocal

reset_globals()
