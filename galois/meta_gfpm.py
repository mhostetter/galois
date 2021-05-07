import numba
import numpy as np

from .dtypes import DTYPES
from .meta_gf import GFMeta
from .poly import Poly

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
DEGREE = None  # The prime power `m` of the Galois field
ORDER = None  # The field's order `p^m`
PRIMITIVE_ELEMENT = None  # The field's primitive element
IRREDUCIBLE_POLY = None  # The field's primitive polynomial in decimal form
DTYPE = None  # The largest valid data-type for this field

# Placeholder functions to be replaced by JIT-compiled function
INT_TO_POLY_JIT = lambda x: [0]
POLY_TO_INT_JIT = lambda vec: 0
ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y
MULTIPLICATIVE_INVERSE_JIT = lambda x: 1 / x


class GFpmMeta(GFMeta):
    """
    An abstract base class for all :math:`\\mathrm{GF}(2^m)` field array classes.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly_int = cls._irreducible_poly.integer
        cls._irreducible_poly_coeffs = np.array(cls._irreducible_poly.coeffs, dtype=cls.dtypes[-1])  # pylint: disable=unsubscriptable-object
        cls._primitive_element_int = cls._primitive_element.integer
        cls._prime_subfield = kwargs["prime_subfield"]

        cls.compile(kwargs["mode"], kwargs["target"])

        # Convert primitive element from a poly in GF(2)[x] to an integer
        cls._primitive_element = cls(cls._primitive_element.integer)

        # Determine if the irreducible polynomial is primitive
        cls._is_primitive_poly = cls._irreducible_poly(cls.primitive_element, field=cls) == 0

    @property
    def dtypes(cls):
        max_dtype = DTYPES[-1]
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(d) == 0:
            d = [np.object_]
        return d

    def _compile_jit_calculate(cls, target):
        global CHARACTERISTIC, DEGREE, ORDER, PRIMITIVE_ELEMENT, IRREDUCIBLE_POLY, DTYPE, INT_TO_POLY_JIT, POLY_TO_INT_JIT, ADD_JIT, MULTIPLY_JIT, MULTIPLICATIVE_INVERSE_JIT
        CHARACTERISTIC = cls.characteristic
        DEGREE = cls.degree
        ORDER = cls.order
        if isinstance(cls._primitive_element, Poly):
            PRIMITIVE_ELEMENT = cls._primitive_element.integer
        else:
            PRIMITIVE_ELEMENT = int(cls._primitive_element)
        IRREDUCIBLE_POLY = cls._irreducible_poly.coeffs.view(np.ndarray)
        DTYPE = cls.dtypes[-1]  # pylint: disable=unsubscriptable-object

        INT_TO_POLY_JIT = numba.jit("int64[:](int64)", nopython=True)(_int_to_poly)
        POLY_TO_INT_JIT = numba.jit("int64(int64[:])", nopython=True)(_poly_to_int)

        # JIT-compile add,  multiply, and multiplicative inverse routines for reference in polynomial evaluation routine
        ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(_add_calculate)
        MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(_multiply_calculate)
        MULTIPLICATIVE_INVERSE_JIT = numba.jit("int64(int64)", nopython=True)(_multiplicative_inverse_calculate)

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # Create numba JIT-compiled ufuncs
        cls._ufuncs["add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_calculate)
        cls._ufuncs["subtract"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_calculate)
        cls._ufuncs["multiply"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_calculate)
        cls._ufuncs["divide"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_calculate)
        cls._ufuncs["negative"] = numba.vectorize(["int64(int64)"], **kwargs)(_additive_inverse_calculate)
        cls._ufuncs["reciprocal"] = numba.vectorize(["int64(int64)"], **kwargs)(_multiplicative_inverse_calculate)
        cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
        cls._ufuncs["log"] = numba.vectorize(["int64(int64)"], **kwargs)(_log_calculate)
        cls._ufuncs["poly_eval"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_calculate)

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

    # def _ufunc_add(cls, ufunc, method, inputs, kwargs, meta):
    #     if cls._ufunc_mode == "jit-lookup":
    #         # Use the lookup ufunc on each array entry
    #         return super()._ufunc_add(ufunc, method, inputs, kwargs, meta)
    #     else:
    #         # Convert to entire polynomial/vector representation and perform array operation in GF(p)
    #         cls._verify_operands_in_same_field(ufunc, inputs, meta)
    #         inputs, kwargs = cls._convert_inputs_to_vector(inputs, kwargs)
    #         output = getattr(ufunc, method)(*inputs, **kwargs)
    #         output = cls._convert_output_from_vector(output, meta["field"], meta["dtype"])
    #         return output

    # def _ufunc_subtract(cls, ufunc, method, inputs, kwargs, meta):
    #     if cls._ufunc_mode == "jit-lookup":
    #         # Use the lookup ufunc on each array entry
    #         return super()._ufunc_subtract(ufunc, method, inputs, kwargs, meta)
    #     else:
    #         # Convert entire array to polynomial/vector representation and perform array operation in GF(p)
    #         cls._verify_operands_in_same_field(ufunc, inputs, meta)
    #         inputs, kwargs = cls._convert_inputs_to_vector(inputs, kwargs)
    #         output = getattr(ufunc, method)(*inputs, **kwargs)
    #         output = cls._convert_output_from_vector(output, meta["field"], meta["dtype"])
    #         return output

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _int_to_poly(cls, a):
        a_vec = np.zeros(cls.degree, dtype=cls.dtypes[-1])  # pylint: disable=unsubscriptable-object
        for i in range(0, cls.degree):
            q = a // cls.characteristic**(cls.degree - 1 - i)
            a_vec[i] = q
            a -= q*cls.characteristic**(cls.degree - 1 - i)
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

    def _subtract_python(cls, a, b):
        a_vec = cls._int_to_poly(a)
        b_vec = cls._int_to_poly(b)
        c_vec = (a_vec - b_vec) % cls.characteristic
        return cls._poly_to_int(c_vec)

    def _multiply_python(cls, a, b):
        a_vec = cls._int_to_poly(a)
        b_vec = cls._int_to_poly(b)

        c_vec = np.zeros(cls.degree, dtype=cls.dtypes[-1])  # pylint: disable=unsubscriptable-object
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

    def _additive_inverse_python(cls, a):
        a_vec = cls._int_to_poly(a)
        c_vec = (-a_vec) % cls.characteristic
        return cls._poly_to_int(c_vec)

    def _multiplicative_inverse_python(cls, a):
        """
        TODO: Replace this with more efficient algorithm

        From Fermat's Little Theorem:
        a^(p^m - 1) = 1 (mod p^m), for a in GF(p^m)

        a * a^-1 = 1
        a * a^-1 = a^(p^m - 1)
            a^-1 = a^(p^m - 2)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        power = cls.order - 2
        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while power > 1:
            if power % 2 == 0:
                result_s = cls._multiply_python(result_s, result_s)
                power //= 2
            else:
                result_m = cls._multiply_python(result_m, result_s)
                power -= 1

        result = cls._multiply_python(result_m, result_s)

        return result


###############################################################################
# Galois field arithmetic, explicitly calculated without lookup tables
###############################################################################

def _int_to_poly(a):
    """
    Convert the integer representation to vector/polynomial representation
    """
    a_vec = np.zeros(DEGREE, dtype=DTYPE)
    for i in range(0, DEGREE):
        q = a // CHARACTERISTIC**(DEGREE - 1 - i)
        a_vec[i] = q
        a -= q*CHARACTERISTIC**(DEGREE - 1 - i)
    return a_vec


def _poly_to_int(a_vec):
    """
    Convert the integer representation to vector/polynomial representation
    """
    a = 0
    for i in range(0, DEGREE):
        a += a_vec[i]*CHARACTERISTIC**(DEGREE - 1 - i)
    return a


def _add_calculate(a, b):  # pragma: no cover
    a_vec = INT_TO_POLY_JIT(a)
    b_vec = INT_TO_POLY_JIT(b)
    c_vec = (a_vec + b_vec) % CHARACTERISTIC
    return POLY_TO_INT_JIT(c_vec)


def _subtract_calculate(a, b):  # pragma: no cover
    a_vec = INT_TO_POLY_JIT(a)
    b_vec = INT_TO_POLY_JIT(b)
    c_vec = (a_vec - b_vec) % CHARACTERISTIC
    return POLY_TO_INT_JIT(c_vec)


def _multiply_calculate(a, b):  # pragma: no cover
    a_vec = INT_TO_POLY_JIT(a)
    b_vec = INT_TO_POLY_JIT(b)

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
            a_vec = (a_vec - q*IRREDUCIBLE_POLY[1:]) % CHARACTERISTIC

        # Divide b(x) by x
        b_vec[1:] = b_vec[:-1]
        b_vec[0] = 0

    return POLY_TO_INT_JIT(c_vec)


def _divide_calculate(a, b):  # pragma: no cover
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:
        return 0
    else:
        b_inv = MULTIPLICATIVE_INVERSE_JIT(b)
        return MULTIPLY_JIT(a, b_inv)


def _additive_inverse_calculate(a):  # pragma: no cover
    a_vec = INT_TO_POLY_JIT(a)
    a_vec = (-a_vec) % CHARACTERISTIC
    return POLY_TO_INT_JIT(a_vec)


def _multiplicative_inverse_calculate(a):  # pragma: no cover
    """
    TODO: Replace this with a more efficient algorithm

    From Fermat's Little Theorem:
    a^(p^m - 1) = 1, for a in GF(p^m)

    a * a^-1 = 1
    a * a^-1 = a^(p^m - 1)
        a^-1 = a^(p^m - 2)
    """
    if a == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    power = ORDER - 2
    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = MULTIPLY_JIT(result_s, result_s)
            power //= 2
        else:
            result_m = MULTIPLY_JIT(result_m, result_s)
            power -= 1

    result = MULTIPLY_JIT(result_m, result_s)

    return result


def _power_calculate(a, power):  # pragma: no cover
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
    if a == 0 and power < 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if power == 0:
        return 1
    elif power < 0:
        a = MULTIPLICATIVE_INVERSE_JIT(a)
        power = abs(power)

    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = MULTIPLY_JIT(result_s, result_s)
            power //= 2
        else:
            result_m = MULTIPLY_JIT(result_m, result_s)
            power -= 1

    result = MULTIPLY_JIT(result_m, result_s)

    return result


def _log_calculate(beta):  # pragma: no cover
    """
    TODO: Replace this with more efficient algorithm

    alpha in GF(p^m) and generates field
    beta in GF(p^m)

    gamma = log_primitive_element(beta), such that: alpha^gamma = beta
    """
    if beta == 0:
        raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

    # Naive algorithm
    result = 1
    for i in range(0, ORDER - 1):
        if result == beta:
            break
        result = MULTIPLY_JIT(result, PRIMITIVE_ELEMENT)

    return i


def _poly_eval_calculate(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
