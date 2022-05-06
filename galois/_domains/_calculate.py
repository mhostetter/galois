"""
A module containing various ufunc dispatchers using explicit calculation.
"""
from typing import Any, Type

import numba
import numpy as np

from . import _lookup
from ._array import Array


###############################################################################
# Helper JIT functions
###############################################################################

DTYPE = np.int64
INT_TO_VECTOR: Any
VECTOR_TO_INT: Any


@numba.jit(["int64[:](int64, int64, int64)"], nopython=True, cache=True)
def int_to_vector(a: int, characteristic: int, degree: int) -> np.ndarray:
    """
    Converts the integer representation to vector/polynomial representation.
    """
    a_vec = np.zeros(degree, dtype=DTYPE)
    for i in range(degree - 1, -1, -1):
        q, r = divmod(a, characteristic)
        a_vec[i] = r
        a = q

    return a_vec


@numba.jit(["int64(int64[:], int64, int64)"], nopython=True, cache=True)
def vector_to_int(a_vec: np.ndarray, characteristic: int, degree: int) -> int:
    """
    Converts the vector/polynomial representation to the integer representation.
    """
    a = 0
    factor = 1
    for i in range(degree - 1, -1, -1):
        a += a_vec[i] * factor
        factor *= characteristic

    return a


def set_helper_globals(field: Type[Array]):
    global DTYPE, INT_TO_VECTOR, VECTOR_TO_INT
    if field.ufunc_mode != "python-calculate":
        DTYPE = np.int64
        INT_TO_VECTOR = int_to_vector
        VECTOR_TO_INT = vector_to_int
    else:
        DTYPE = np.object_
        INT_TO_VECTOR = int_to_vector.py_func
        VECTOR_TO_INT = vector_to_int.py_func


class add_modular(_lookup.add_ufunc):
    """
    A ufunc dispatcher that provides addition modulo the characteristic.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int, b: int) -> int:
        c = a + b
        if c >= CHARACTERISTIC:
            c -= CHARACTERISTIC
        return c


class add_vector(_lookup.add_ufunc):
    """
    A ufunc dispatcher that provides addition for extensions.
    """
    def __call__(self, ufunc, method, inputs, kwargs, meta):
        if self.field.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super().__call__(ufunc, method, inputs, kwargs, meta)
        else:
            # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert back to GF(p^m)
            self._verify_operands_in_same_field(ufunc, inputs, meta)
            inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
            output = getattr(ufunc, method)(*inputs, **kwargs)
            output = self._convert_output_from_vector(output, meta["dtype"])
            return output

    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC, DEGREE
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int, b: int) -> int:
        a_vec = INT_TO_VECTOR(a, CHARACTERISTIC, DEGREE)
        b_vec = INT_TO_VECTOR(b, CHARACTERISTIC, DEGREE)
        c_vec = (a_vec + b_vec) % CHARACTERISTIC
        c = VECTOR_TO_INT(c_vec, CHARACTERISTIC, DEGREE)

        return c


class negative_modular(_lookup.negative_ufunc):
    """
    A ufunc dispatcher that provides additive inverse modulo the characteristic.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int) -> int:
        if a == 0:
            c = 0
        else:
            c = CHARACTERISTIC - a
        return c


class negative_vector(_lookup.negative_ufunc):
    """
    A ufunc dispatcher that provides additive inverse for extensions.
    """
    def __call__(self, ufunc, method, inputs, kwargs, meta):
        if self.field.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super().__call__(ufunc, method, inputs, kwargs, meta)
        else:
            # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert back to GF(p^m)
            self._verify_operands_in_same_field(ufunc, inputs, meta)
            inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
            output = getattr(ufunc, method)(*inputs, **kwargs)
            output = self._convert_output_from_vector(output, meta["dtype"])
            return output

    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC, DEGREE
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int) -> int:
        a_vec = INT_TO_VECTOR(a, CHARACTERISTIC, DEGREE)
        c_vec = (-a_vec) % CHARACTERISTIC
        c = VECTOR_TO_INT(c_vec, CHARACTERISTIC, DEGREE)

        return c


class subtract_modular(_lookup.subtract_ufunc):
    """
    A ufunc dispatcher that provides subtraction modulo the characteristic.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int, b: int) -> int:
        if a >= b:
            c = a - b
        else:
            c = CHARACTERISTIC + a - b

        return c


class subtract_vector(_lookup.subtract_ufunc):
    """
    A ufunc dispatcher that provides subtraction for extensions.
    """
    def __call__(self, ufunc, method, inputs, kwargs, meta):
        if self.field.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super().__call__(ufunc, method, inputs, kwargs, meta)
        else:
            # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert back to GF(p^m)
            self._verify_operands_in_same_field(ufunc, inputs, meta)
            inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
            output = getattr(ufunc, method)(*inputs, **kwargs)
            output = self._convert_output_from_vector(output, meta["dtype"])
            return output

    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC, DEGREE
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int, b: int) -> int:
        a_vec = INT_TO_VECTOR(a, CHARACTERISTIC, DEGREE)
        b_vec = INT_TO_VECTOR(b, CHARACTERISTIC, DEGREE)
        c_vec = (a_vec - b_vec) % CHARACTERISTIC
        c = VECTOR_TO_INT(c_vec, CHARACTERISTIC, DEGREE)

        return c


class multiply_binary(_lookup.multiply_ufunc):
    """
    A ufunc dispatcher that provides multiplication modulo 2.

    Algorithm:
        a in GF(2^m), can be represented as a degree m-1 polynomial a(x) in GF(2)[x]
        b in GF(2^m), can be represented as a degree m-1 polynomial b(x) in GF(2)[x]
        p(x) in GF(2)[x] with degree m is the irreducible polynomial of GF(2^m)

        a * b = c
              = (a(x) * b(x)) % p(x) in GF(2)
              = c(x)
              = c
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global ORDER, IRREDUCIBLE_POLY
        ORDER = self.field.order
        IRREDUCIBLE_POLY = self.field._irreducible_poly_int

    @staticmethod
    def calculate(a: int, b: int) -> int:
        # Re-order operands such that a > b so the while loop has less loops
        if b > a:
            a, b = b, a

        c = 0
        while b > 0:
            if b & 0b1:
                c ^= a  # Add a(x) to c(x)

            b >>= 1  # Divide b(x) by x
            a <<= 1  # Multiply a(x) by x
            if a >= ORDER:
                a ^= IRREDUCIBLE_POLY  # Compute a(x) % p(x)

        return c


class multiply_modular(_lookup.multiply_ufunc):
    """
    A ufunc dispatcher that provides multiplication modulo the characteristic.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int, b: int) -> int:
        c = (a * b) % CHARACTERISTIC

        return c


class multiply_vector(_lookup.multiply_ufunc):
    """
    A ufunc dispatcher that provides multiplication for extensions.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        IRREDUCIBLE_POLY = self.field._irreducible_poly_int
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int, b: int) -> int:
        a_vec = INT_TO_VECTOR(a, CHARACTERISTIC, DEGREE)
        b_vec = INT_TO_VECTOR(b, CHARACTERISTIC, DEGREE)

        # The irreducible polynomial with the x^degree term removed
        irreducible_poly_vec = INT_TO_VECTOR(IRREDUCIBLE_POLY - CHARACTERISTIC**DEGREE, CHARACTERISTIC, DEGREE)

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

        c = VECTOR_TO_INT(c_vec, CHARACTERISTIC, DEGREE)

        return c


class reciprocal_modular_egcd(_lookup.reciprocal_ufunc):
    """
    A ufunc dispatcher that provides the multiplicative inverse modulo the characteristic.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int) -> int:
        """
        s*x + t*y = gcd(x, y) = 1
        x = p
        y = a in GF(p)
        t = a**-1 in GF(p)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        r2, r1 = CHARACTERISTIC, a
        t2, t1 = 0, 1

        while r1 != 0:
            q = r2 // r1
            r2, r1 = r1, r2 - q*r1
            t2, t1 = t1, t2 - q*t1

        if t2 < 0:
            t2 += CHARACTERISTIC

        return t2


class reciprocal_fermat(_lookup.reciprocal_ufunc):
    """
    A ufunc dispatcher that provides the multiplicative inverse using Fermat's Little Theorem.

    Algorithm:
        a in GF(p^m)
        a^(p^m - 1) = 1

        a * a^-1 = 1
        a * a^-1 = a^(p^m - 1)
            a^-1 = a^(p^m - 2)
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global ORDER, MULTIPLY
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc

    @staticmethod
    def calculate(a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        exponent = ORDER - 2
        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while exponent > 1:
            if exponent % 2 == 0:
                result_s = MULTIPLY(result_s, result_s)
                exponent //= 2
            else:
                result_m = MULTIPLY(result_m, result_s)
                exponent -= 1

        result = MULTIPLY(result_m, result_s)

        return result


class divide(_lookup.divide_ufunc):
    """
    A ufunc dispatcher that provides division.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global MULTIPLY, RECIPROCAL
        MULTIPLY = self.field._multiply.ufunc
        RECIPROCAL = self.field._reciprocal.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            c = 0
        else:
            b_inv = RECIPROCAL(b)
            c = MULTIPLY(a, b_inv)

        return c


class field_power_square_and_multiply(_lookup.power_ufunc):
    """
    A ufunc dispatcher that provides exponentiation using the Square and Multiply algorithm.

    - This algorithm is applicable to fields since the exponent may be negative.

    Algorithm:
        a^13 = (1) * (a)^13
             = (a) * (a)^12
             = (a) * (a^2)^6
             = (a) * (a^4)^3
             = (a * a^4) * (a^4)^2
             = (a * a^4) * (a^8)
             = result_m * result_s
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global MULTIPLY, RECIPROCAL
        MULTIPLY = self.field._multiply.ufunc
        RECIPROCAL = self.field._reciprocal.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif b < 0:
            a = RECIPROCAL(a)
            b = abs(b)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while b > 1:
            if b % 2 == 0:
                result_s = MULTIPLY(result_s, result_s)
                b //= 2
            else:
                result_m = MULTIPLY(result_m, result_s)
                b -= 1

        result = MULTIPLY(result_m, result_s)

        return result


class log_brute_force(_lookup.log_ufunc):
    """
    A ufunc dispatcher that provides logarithm calculation using a brute-force search.
    """
    def set_calculate_globals(self):
        # pylint: disable=global-variable-undefined
        global ORDER, MULTIPLY
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """
        a = α^m
        b is a primitive element of the field

        c = log(a, b)
        a = b^c
        """
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        # Naive algorithm
        result = 1
        for i in range(0, ORDER - 1):
            if result == a:
                break
            result = MULTIPLY(result, b)

        return i


class sqrt_binary(_lookup.sqrt_ufunc):
    """
    A ufunc dispatcher that provides the square root in binary extension fields.
    """
    def implementation(self, a: Array) -> Array:
        """
        Fact 3.42 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        return a ** (self.field.characteristic**(self.field.degree - 1))


class sqrt(_lookup.sqrt_ufunc):
    """
    A ufunc dispatcher that provides square root using NumPy array arithmetic.
    """
    def implementation(self, a: Array) -> Array:
        """
        Algorithm 3.34 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        Algorithm 3.36 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        p = self.field.characteristic

        if p % 4 == 3:
            roots = a ** ((self.field.order + 1)//4)

        elif p % 8 == 5:
            d = a ** ((self.field.order - 1)//4)
            roots = self.field.Zeros(a.shape)

            idxs = np.where(d == 1)
            roots[idxs] = a[idxs] ** ((self.field.order + 3)//8)

            idxs = np.where(d == p - 1)
            roots[idxs] = 2*a[idxs] * (4*a[idxs]) ** ((self.field.order - 5)//8)

        else:
            # Find a quadratic non-residue element `b`
            while True:
                b = self.field.Random(low=1)
                if not b.is_quadratic_residue():
                    break

            # Write p - 1 = 2^s * t
            n = self.field.order - 1
            s = 0
            while n % 2 == 0:
                n >>= 1
                s += 1
            t = n
            assert self.field.order - 1 == 2**s * t

            roots = self.field.Zeros(a.shape)  # Empty array of roots

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

        roots = self.field._view(np.minimum(roots, -roots))  # Return only the smaller root

        return roots
