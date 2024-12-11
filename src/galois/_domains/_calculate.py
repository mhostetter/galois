"""
A module containing various ufunc dispatchers with explicit calculation arithmetic added. Various algorithms for
each type of arithmetic are implemented here.
"""

from typing import Type

import numba
import numpy as np

from .._prime import factors
from . import _lookup
from ._array import Array

###############################################################################
# Helper JIT functions
###############################################################################

DTYPE = np.int64


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


@numba.jit(["int64[:](int64, int64)"], nopython=True, cache=True)
def egcd(a: int, b: int) -> np.ndarray:  # pragma: no cover
    """
    Computes the Extended Euclidean Algorithm. Returns (d, s, t).

    Algorithm:
        s*x + t*y = gcd(x, y) = d
    """
    r2, r1 = a, b
    s2, s1 = 1, 0
    t2, t1 = 0, 1

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q * r1
        s2, s1 = s1, s2 - q * s1
        t2, t1 = t1, t2 - q * t1

    # Ensure the GCD is positive
    if r2 < 0:
        r2 *= -1
        s2 *= -1
        t2 *= -1

    return np.array([r2, s2, t2], dtype=DTYPE)


EGCD = egcd


@numba.jit(["int64(int64[:], int64[:])"], nopython=True, cache=True)
def crt(remainders: np.ndarray, moduli: np.ndarray) -> int:  # pragma: no cover
    """
    Computes the simultaneous solution to the system of congruences xi == ai (mod mi).
    """
    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1, m1 = remainders[0], moduli[0]
    for a2, m2 in zip(remainders[1:], moduli[1:]):
        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = gcd(m1, m2).
        d, b1, b2 = EGCD(m1, m2)

        if d == 1:
            # The moduli (m1, m2) are coprime
            x = (a1 * b2 * m2) + (a2 * b1 * m1)  # Compute x through explicit construction
            m1 = m1 * m2  # The new modulus
        else:
            # The moduli (m1, m2) are not coprime, however if a1 == b2 (mod d)
            # then a unique solution still exists.
            if not (a1 % d) == (a2 % d):
                raise ArithmeticError
            x = ((a1 * b2 * m2) + (a2 * b1 * m1)) // d  # Compute x through explicit construction
            m1 = (m1 * m2) // d  # The new modulus

        a1 = x % m1  # The new equivalent remainder

    # At the end of the process x == a1 (mod m1) where a1 and m1 are the new/modified residual
    # and remainder.

    return a1


def set_helper_globals(field: Type[Array]):
    global DTYPE, INT_TO_VECTOR, VECTOR_TO_INT, EGCD, CRT
    if field.ufunc_mode != "python-calculate":
        DTYPE = np.int64
        INT_TO_VECTOR = int_to_vector
        VECTOR_TO_INT = vector_to_int
        EGCD = egcd
        CRT = crt
    else:
        DTYPE = np.object_
        INT_TO_VECTOR = int_to_vector.py_func
        VECTOR_TO_INT = vector_to_int.py_func
        EGCD = egcd.py_func
        CRT = crt.py_func


###############################################################################
# Specific explicit calculation algorithms
###############################################################################


class add_modular(_lookup.add_ufunc):
    """
    A ufunc dispatcher that provides addition modulo the characteristic.
    """

    def set_calculate_globals(self):
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

        # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert
        # back to GF(p^m).
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
        output = getattr(ufunc, method)(*inputs, **kwargs)
        output = self._convert_output_from_vector(output, meta["dtype"])
        return output

    def set_calculate_globals(self):
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

        # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert
        # back to GF(p^m).
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
        output = getattr(ufunc, method)(*inputs, **kwargs)
        output = self._convert_output_from_vector(output, meta["dtype"])
        return output

    def set_calculate_globals(self):
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

        # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert
        # back to GF(p^m).
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
        output = getattr(ufunc, method)(*inputs, **kwargs)
        output = self._convert_output_from_vector(output, meta["dtype"])
        return output

    def set_calculate_globals(self):
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
                c_vec = (c_vec + b_vec[-1] * a_vec) % CHARACTERISTIC

            # Multiply a(x) by x
            q = a_vec[0]
            a_vec[:-1] = a_vec[1:]
            a_vec[-1] = 0

            # Reduce a(x) modulo the irreducible polynomial
            if q > 0:
                a_vec = (a_vec - q * irreducible_poly_vec) % CHARACTERISTIC

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
            r2, r1 = r1, r2 - q * r1
            t2, t1 = t1, t2 - q * t1

        if t2 < 0:
            t2 += CHARACTERISTIC

        return t2


# NOTE: Commented out because it's not currently being used. This prevents it from being
#       flagged as "not covered".
# class reciprocal_fermat(_lookup.reciprocal_ufunc):
#     """
#     A ufunc dispatcher that provides the multiplicative inverse using Fermat's Little Theorem.

#     Algorithm:
#         a in GF(p^m)
#         a^(p^m - 1) = 1

#         a * a^-1 = 1
#         a * a^-1 = a^(p^m - 1)
#             a^-1 = a^(p^m - 2)
#     """
#     def set_calculate_globals(self):
#         global ORDER, POSITIVE_POWER
#         ORDER = self.field.order
#         POSITIVE_POWER = self.field._positive_power.ufunc

#     @staticmethod
#     def calculate(a: int) -> int:
#         if a == 0:
#             raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

#         return POSITIVE_POWER(a, ORDER - 2)


class reciprocal_itoh_tsujii(_lookup.reciprocal_ufunc):
    """
    A ufunc dispatcher that provides the multiplicative inverse using the Itoh-Tsujii inversion algorithm.

    Algorithm:
        a in GF(p^m)

        1. Compute r = (p^m - 1) / (p - 1)
        2. Compute a^(r - 1) in GF(p^m)
        3. Compute a^r = a^(r - 1) * a = a.field_norm(), a^r is in GF(p)
        4. Compute (a^r)^-1 in GF(p)
        5. Compute a^-1 = (a^r)^-1 * a^(r - 1)
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC, ORDER, MULTIPLY, POSITIVE_POWER, SUBFIELD_RECIPROCAL
        CHARACTERISTIC = self.field.characteristic
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc
        POSITIVE_POWER = self.field._positive_power.ufunc
        SUBFIELD_RECIPROCAL = getattr(self.field.prime_subfield._reciprocal, self.field.ufunc_mode.replace("-", "_"))

    @staticmethod
    def calculate(a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        # Step 1: Compute r = (p^m - 1) / (p - 1)
        r = (ORDER - 1) // (CHARACTERISTIC - 1)

        # Step 2: Compute a^(r - 1)
        a_r1 = POSITIVE_POWER(a, r - 1)

        # Step 3: Compute a^r = a^(r - 1) * a, a^r is in GF(p)
        a_r = MULTIPLY(a_r1, a)

        # Step 4: Compute (a^r)^-1 in GF(p)
        a_r_inv = SUBFIELD_RECIPROCAL(a_r)

        # Step 5: Compute a^-1 = (a^r)^-1 * a^(r - 1)
        a_inv = MULTIPLY(a_r_inv, a_r1)

        return a_inv


class divide(_lookup.divide_ufunc):
    """
    A ufunc dispatcher that provides division.
    """

    def set_calculate_globals(self):
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


class positive_power_square_and_multiply(_lookup.power_ufunc):
    """
    A ufunc dispatcher that provides exponentiation (positive exponents only) using the Square and Multiply algorithm.

    Algorithm:
        a^13 = (1) * (a)^13
             = (a) * (a)^12
             = (a) * (a^2)^6
             = (a) * (a^4)^3
             = (a * a^4) * (a^4)^2
             = (a * a^4) * (a^8)
           c = c_m * c_s
    """

    def set_calculate_globals(self):
        global MULTIPLY
        MULTIPLY = self.field._multiply.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        assert b >= 0

        if b == 0:
            return 1

        c_square = a  # The "squaring" part
        c_mult = 1  # The "multiplicative" part

        while b > 1:
            if b % 2 == 0:
                c_square = MULTIPLY(c_square, c_square)
                b //= 2
            else:
                c_mult = MULTIPLY(c_mult, c_square)
                b -= 1
        c = MULTIPLY(c_mult, c_square)

        return c


class power_square_and_multiply(_lookup.power_ufunc):
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
        global RECIPROCAL, POSITIVE_POWER
        RECIPROCAL = self.field._reciprocal.ufunc
        POSITIVE_POWER = self.field._positive_power.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            c = 1
        elif b > 0:
            c = POSITIVE_POWER(a, b)
        else:
            a_inv = RECIPROCAL(a)
            c = POSITIVE_POWER(a_inv, abs(b))

        return c


class log_brute_force(_lookup.log_ufunc):
    """
    A ufunc dispatcher that provides logarithm calculation using a brute-force search.
    """

    def set_calculate_globals(self):
        global ORDER, MULTIPLY
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc

    @staticmethod
    def calculate(beta: int, alpha: int) -> int:  # pragma: no cover
        """
        beta is an element of GF(p^m)
        alpha is a primitive element of GF(p^m)

        i = log(beta, alpha)
        beta = alpha^i
        """
        if beta == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        c = 1
        for i in range(0, ORDER - 1):
            if c == beta:
                return i
            c = MULTIPLY(c, alpha)

        raise ArithmeticError("The specified logarithm base is not a primitive element of the Galois field.")


class log_pollard_rho(_lookup.log_ufunc):
    """
    A ufunc dispatcher that provides logarithm calculation using the Pollard ρ algorithm.
    """

    def set_calculate_globals(self):
        global ORDER, MULTIPLY
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc
        set_helper_globals(self.field)

    @staticmethod
    def calculate(beta: int, alpha: int) -> int:  # pragma: no cover
        """
        beta is an element of GF(p^m)
        alpha is a primitive element of GF(p^m)
        Compute x = log_alpha(beta)

        Algorithm 3.60 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf
        """
        if beta == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        n = ORDER - 1  # Order of the multiplicative group of GF(p^m), must be prime
        x0, a0, b0 = 1, 0, 0
        xi, ai, bi = x0, a0, b0
        x2i, a2i, b2i = xi, ai, bi

        def compute_x(x):
            # Equation 3.2
            if x % 3 == 1:
                return MULTIPLY(beta, x)
            if x % 3 == 2:
                return MULTIPLY(x, x)
            return MULTIPLY(alpha, x)

        def compute_a(a, x):
            # Equation 3.3
            if x % 3 == 1:
                return a
            if x % 3 == 2:
                return (2 * a) % n
            return (a + 1) % n

        def compute_b(b, x):
            # Equation 3.4
            if x % 3 == 1:
                return (b + 1) % n
            if x % 3 == 2:
                return (2 * b) % n
            return b

        while True:
            xi, ai, bi = compute_x(xi), compute_a(ai, xi), compute_b(bi, xi)

            x2i, a2i, b2i = compute_x(x2i), compute_a(a2i, x2i), compute_b(b2i, x2i)
            x2i, a2i, b2i = compute_x(x2i), compute_a(a2i, x2i), compute_b(b2i, x2i)

            if xi == x2i:
                r = (bi - b2i) % n
                if r != 0:
                    d, r_inv = EGCD(r, n)[0:2]
                    assert d == 1
                    return (r_inv * (a2i - ai)) % n

                # Re-try with different x0, a0, and b0
                a0 += 1
                b0 += 1
                x0 = MULTIPLY(x0, beta)
                x0 = MULTIPLY(x0, alpha)
                xi, ai, bi = x0, a0, b0
                x2i, a2i, b2i = xi, ai, bi


class log_pohlig_hellman(_lookup.log_ufunc):
    """
    A ufunc dispatcher that provides logarithm calculation using the Pohlig-Hellman algorithm.
    """

    def set_calculate_globals(self):
        global ORDER, MULTIPLY, RECIPROCAL, POWER, BRUTE_FORCE_LOG, FACTORS, MULTIPLICITIES
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc
        RECIPROCAL = self.field._reciprocal.ufunc
        POWER = self.field._power.ufunc
        if self.field.ufunc_mode in ["jit-lookup", "jit-calculate"]:
            # We can never use the lookup table version of log because it has a fixed base
            BRUTE_FORCE_LOG = log_brute_force(self.field).jit_calculate
        else:
            BRUTE_FORCE_LOG = log_brute_force(self.field).python_calculate
        FACTORS, MULTIPLICITIES = factors(self.field.order - 1)
        set_helper_globals(self.field)
        FACTORS = np.array(FACTORS, dtype=DTYPE)
        MULTIPLICITIES = np.array(MULTIPLICITIES, dtype=DTYPE)

    @staticmethod
    def calculate(beta: int, alpha: int) -> int:  # pragma: no cover
        """
        beta is an element of GF(p^m)
        alpha is a primitive element of GF(p^m)
        The n = p1^e1 * ... * pr^er prime factorization is required
        Compute x = log_alpha(beta)

        Algorithm 3.63 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf
        """
        if beta == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        r = len(FACTORS)
        n = ORDER - 1  # Order of the multiplicative group of GF(p^m), must be prime

        x = np.zeros(r, dtype=DTYPE)
        m = np.zeros(r, dtype=DTYPE)
        for i in range(r):
            q = FACTORS[i]
            e = MULTIPLICITIES[i]
            m[i] = q**e
            gamma = 1
            alpha_bar = POWER(alpha, n // q)
            l_prev = 0  # Starts as l_i-1
            q_prev = 0  # Starts as q^(-1)
            for j in range(e):
                gamma = MULTIPLY(gamma, POWER(alpha, l_prev * q_prev))
                beta_bar = POWER(MULTIPLY(beta, RECIPROCAL(gamma)), n // (q ** (j + 1)))
                l = BRUTE_FORCE_LOG(beta_bar, alpha_bar)
                x[i] += l * q**j
                l_prev = l
                q_prev = q**j

        return CRT(x, m)


class sqrt_binary(_lookup.sqrt_ufunc):
    """
    A ufunc dispatcher that provides the square root in binary extension fields.
    """

    def implementation(self, a: Array) -> Array:
        """
        Fact 3.42 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        return a ** (self.field.characteristic ** (self.field.degree - 1))


class sqrt(_lookup.sqrt_ufunc):
    """
    A ufunc dispatcher that provides square root using NumPy array arithmetic.
    """

    def implementation(self, a: Array) -> Array:
        """
        Algorithm 3.34 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        Algorithm 3.36 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        if not np.all(a.is_square()):
            raise ArithmeticError(
                f"Input array has elements that are non-squares in {self.field.name}.\n{a[~a.is_square()]}"
            )

        p = self.field.characteristic
        q = self.field.order

        if q % 4 == 3:
            roots = a ** ((q + 1) // 4)

        elif q % 8 == 5:
            d = a ** ((q - 1) // 4)
            roots = self.field.Zeros(a.shape)

            idxs = np.where(d == 1)
            roots[idxs] = a[idxs] ** ((q + 3) // 8)

            idxs = np.where(d == p - 1)
            roots[idxs] = 2 * a[idxs] * (4 * a[idxs]) ** ((q - 5) // 8)

        else:
            # Find a non-square element `b`
            while True:
                b = self.field.Random(low=1)
                if not b.is_square():
                    break

            # Write q - 1 = 2^s * t
            n = q - 1
            s = 0
            while n % 2 == 0:
                n >>= 1
                s += 1
            t = n
            assert q - 1 == 2**s * t

            roots = self.field.Zeros(a.shape)  # Empty array of roots

            # Compute a root `r` for the non-zero elements
            idxs = np.where(a > 0)  # Indices where a has a reciprocal
            a_inv = np.reciprocal(a[idxs])
            c = b**t
            r = a[idxs] ** ((t + 1) // 2)
            for i in range(1, s):
                d = (r**2 * a_inv) ** (2 ** (s - i - 1))
                r[np.where(d == p - 1)] *= c
                c = c**2
            roots[idxs] = r  # Assign non-zero roots to the original array

        roots = self.field._view(np.minimum(roots, -roots))  # Return only the smaller root

        return roots
