"""
A module containing polynomial arithmetic for polynomials with dense coefficients.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import int64

from .._domains import Array
from .._domains._function import Function
from .._helper import verify_isinstance


class add_jit(Function):
    """
    Computes polynomial addition of two polynomials.

    Algorithm:
        c(x) = a(x) + b(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            r = self.jit(a.astype(np.int64), b.astype(np.int64))
            r = r.astype(dtype)
        else:
            r = self.python(a.view(np.ndarray), b.view(np.ndarray))
        r = self.field._view(r)

        return r

    def set_globals(self):
        global ADD
        ADD = self.field._add.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        dtype = a.dtype
        c = np.zeros(max(a.size, b.size), dtype=dtype)
        c[-a.size :] = a
        c[-b.size :] = ADD(c[-b.size :], b)
        return c


def negative(a: Array) -> Array:
    """
    c(x) = -a(x)
    a(x) + -a(x) = 0
    """
    return -a


def subtract(a: Array, b: Array) -> Array:
    """
    c(x) = a(x) - b(x)
    """
    field = type(a)

    # c(x) = a(x) - b(x)
    c = field.Zeros(max(a.size, b.size))
    c[-a.size :] = a
    c[-b.size :] -= b

    return c


class subtract_jit(Function):
    """
    Computes polynomial subtraction of two polynomials.

    Algorithm:
        c(x) = a(x) - b(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            r = self.jit(a.astype(np.int64), b.astype(np.int64))
            r = r.astype(dtype)
        else:
            r = self.python(a.view(np.ndarray), b.view(np.ndarray))
        r = self.field._view(r)

        return r

    def set_globals(self):
        global SUBTRACT
        SUBTRACT = self.field._subtract.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        dtype = a.dtype
        c = np.zeros(max(a.size, b.size), dtype=dtype)
        c[-a.size :] = a
        c[-b.size :] = SUBTRACT(c[-b.size :], b)
        return c


def multiply(a: Array, b: Array) -> Array:
    """
    c(x) = a(x) * b(x)
    c(x) = a(x) * b = a(x) + ... + a(x)
    """
    # c(x) = a(x) * b(x)
    if a.ndim == 0 or b.ndim == 0:
        return a * b

    return np.convolve(a, b)


class divmod_jit(Function):
    """
    Computes polynomial division with remainder of two polynomials.

    Algorithm:
        a(x) = q(x)*b(x) + r(x)
    """

    def __call__(self, a: Array, b: Array) -> tuple[Array, Array]:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)

        a_degree = a.size - 1
        b_degree = b.size - 1

        # TODO: Merge all of this into `implementation()`
        if b_degree == 0:
            q, r = a // b, self.field([0])
        elif a_degree == 0 and a[0] == 0:
            q, r = self.field([0]), self.field([0])
        elif a_degree < b_degree:
            q, r = self.field([0]), a
        else:
            assert 1 <= a.ndim <= 2 and b.ndim == 1
            dtype = a.dtype
            a_1d = a.ndim == 1
            a = np.atleast_2d(a)
            # TODO: Do not support 2D -- it is no longer needed

            q_degree = a.shape[-1] - b.shape[-1]
            r_degree = b.shape[-1] - 1

            if self.field.ufunc_mode != "python-calculate":
                qr = self.jit(a.astype(np.int64), b.astype(np.int64))
                qr = qr.astype(dtype)
            else:
                qr = self.python(a.view(np.ndarray), b.view(np.ndarray))
            qr = self.field._view(qr)

            q = qr[:, 0 : q_degree + 1]
            r = qr[:, q_degree + 1 : q_degree + 1 + r_degree + 1]

            if a_1d:
                q = q.reshape(q.size)
                r = r.reshape(r.size)

        return q, r

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:, :](int64[:, :], int64[:]))

    @staticmethod
    def implementation(a, b):
        assert a.ndim == 2 and b.ndim == 1
        assert a.shape[-1] >= b.shape[-1]

        q_degree = a.shape[1] - b.shape[-1]
        qr = a.copy()

        for k in range(a.shape[0]):
            for i in range(q_degree + 1):
                if qr[k, i] > 0:
                    q = MULTIPLY(qr[k, i], RECIPROCAL(b[0]))
                    for j in range(1, b.size):
                        qr[k, i + j] = SUBTRACT(qr[k, i + j], MULTIPLY(q, b[j]))
                    qr[k, i] = q

        return qr


class floordiv_jit(Function):
    """
    Computes polynomial division without remainder of two polynomials.

    Algorithm:
        a(x) = q(x)*b(x) + r(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            q = self.jit(a.astype(np.int64), b.astype(np.int64))
            q = q.astype(dtype)
        else:
            q = self.python(a.view(np.ndarray), b.view(np.ndarray))
        q = self.field._view(q)

        return q

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        if b.size == 1 and b[0] == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")

        if a.size < b.size:
            return np.array([0], dtype=a.dtype)

        q_degree = a.size - b.size
        q = np.zeros(q_degree + 1, dtype=a.dtype)
        aa = a[0 : q_degree + 1].copy()

        for i in range(q_degree + 1):
            if aa[i] > 0:
                q[i] = MULTIPLY(aa[i], RECIPROCAL(b[0]))
                N = min(b.size, q_degree + 1 - i)  # We don't need to subtract in the "remainder" range
                for j in range(1, N):
                    aa[i + j] = SUBTRACT(aa[i + j], MULTIPLY(q[i], b[j]))

        return q


class mod_jit(Function):
    """
    Computes the modular division of two polynomials.

    Algorithm:
        a(x) = q(x)*b(x) + r(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            r = self.jit(a.astype(np.int64), b.astype(np.int64))
            r = r.astype(dtype)
        else:
            r = self.python(a.view(np.ndarray), b.view(np.ndarray))
        r = self.field._view(r)

        return r

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        if b.size == 1 and b[0] == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")

        if a.size < b.size:
            return a.copy()

        if b.size == 1:
            return np.array([0], dtype=a.dtype)

        q_degree = a.size - b.size
        r_degree = b.size - 1
        r = np.zeros(r_degree + 1, dtype=a.dtype)
        r[1:] = a[0:r_degree]

        for i in range(q_degree + 1):
            r = np.roll(r, -1)
            r[-1] = a[i + r_degree]

            if r[0] > 0:
                q = MULTIPLY(r[0], RECIPROCAL(b[0]))
                for j in range(1, b.size):
                    r[j] = SUBTRACT(r[j], MULTIPLY(q, b[j]))

        r = r[1:]

        # Trim leading zeros to reduce computations in future calls
        if r.size > 1:
            idxs = np.nonzero(r)[0]
            if idxs.size > 0:
                r = r[idxs[0] :]
            else:
                r = r[-1:]

        return r


class pow_jit(Function):
    """
    Performs modular exponentiation on the polynomial f(x).

    Algorithm:
        d(x) = a(x)^b % c(x)
    """

    def __call__(self, a: Array, b: int, c: Array | None = None) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, int)
        verify_isinstance(c, self.field, optional=True)
        assert a.ndim == 1 and c.ndim == 1 if c is not None else True
        dtype = a.dtype

        # Convert the integer b into a vector of int64 [MSWord, ..., LSWord] so arbitrarily large exponents may be
        # passed into the JIT-compiled version. Each element of b_vec is a 63-bit word.
        b_vec = []  # Pop on LSWord -> MSWord
        while b >= 2**63:
            q, r = divmod(b, 2**63)
            b_vec.append(r)
            b = q
        b_vec.append(b)
        b_vec = np.array(b_vec[::-1], dtype=np.int64)  # Make vector MSWord -> LSWord

        if self.field.ufunc_mode != "python-calculate":
            c_ = np.array([], dtype=np.int64) if c is None else c.astype(np.int64)
            z = self.jit(a.astype(np.int64), b_vec, c_)
            z = z.astype(dtype)
        else:
            c_ = np.array([], dtype=dtype) if c is None else c.view(np.ndarray)
            z = self.python(a.view(np.ndarray), b_vec, c_)
        z = self.field._view(z)

        return z

    def set_globals(self):
        global POLY_MULTIPLY, POLY_MOD
        POLY_MULTIPLY = self.field._convolve.function
        POLY_MOD = mod_jit(self.field).function

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:], int64[:]))

    @staticmethod
    def implementation(a, b_vec, c):
        """
        b is a vector of int64 [MSWord, ..., LSWord] so that arbitrarily large exponents may be passed
        """
        if b_vec.size == 1 and b_vec[0] == 0:
            return np.array([1], dtype=a.dtype)

        result_s = a.copy()  # The "squaring" part
        result_m = np.array([1], dtype=a.dtype)  # The "multiplicative" part

        # Loop from LSWord to MSWord
        for i in range(b_vec.size - 1, -1, -1):
            j = 0  # Bit counter -- make sure we iterate through 63 bits on all but the most-significant word
            while j < 63:
                if i == 0 and b_vec[i] <= 1:
                    # This is the MSB and we already accounted for the most-significant bit -- can exit now
                    break

                if b_vec[i] % 2 == 0:
                    result_s = POLY_MULTIPLY(result_s, result_s)
                    if c.size > 0:
                        result_s = POLY_MOD(result_s, c)
                    b_vec[i] //= 2
                    j += 1
                else:
                    result_m = POLY_MULTIPLY(result_m, result_s)
                    if c.size > 0:
                        result_m = POLY_MOD(result_m, c)
                    b_vec[i] -= 1

        result = POLY_MULTIPLY(result_s, result_m)
        if c.size > 0:
            result = POLY_MOD(result, c)

        return result


class evaluate_elementwise_jit(Function):
    """
    Evaluates the polynomial f(x) elementwise at xi.
    """

    def __call__(self, coeffs: Array, x: Array) -> Array:
        dtype = x.dtype
        shape = x.shape
        x = np.atleast_1d(x.flatten())

        if self.field.ufunc_mode != "python-calculate":
            y = self.jit(coeffs.astype(np.int64), x.astype(np.int64))
            y = y.astype(dtype)
        else:
            y = self.python(coeffs.view(np.ndarray), x.view(np.ndarray))
        y = self.field._view(y)
        y = y.reshape(shape)

        return y

    def set_globals(self):
        global ADD, MULTIPLY
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))
    _PARALLEL = True

    @staticmethod
    def implementation(coeffs, values):
        y = np.zeros(values.size, dtype=values.dtype)
        for i in numba.prange(values.size):
            y[i] = coeffs[0]
            for j in range(1, coeffs.size):
                y[i] = ADD(coeffs[j], MULTIPLY(y[i], values[i]))

        return y


class roots_jit(Function):
    """
    Finds the roots of the polynomial f(x).
    """

    def __call__(self, nonzero_degrees: np.ndarray, nonzero_coeffs: Array) -> Array:
        verify_isinstance(nonzero_degrees, np.ndarray)
        verify_isinstance(nonzero_coeffs, self.field)
        dtype = nonzero_coeffs.dtype

        if self.field.ufunc_mode != "python-calculate":
            roots = self.jit(
                nonzero_degrees.astype(np.int64), nonzero_coeffs.astype(np.int64), int(self.field.primitive_element)
            )[0, :]
            roots = roots.astype(dtype)
        else:
            roots = self.python(
                nonzero_degrees.view(np.ndarray), nonzero_coeffs.view(np.ndarray), int(self.field.primitive_element)
            )[0, :]
        roots = self.field._view(roots)
        idxs = np.argsort(roots)

        return roots[idxs]

    def set_globals(self):
        global ORDER, ADD, MULTIPLY, POWER
        ORDER = self.field.order
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        POWER = self.field._power.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:, :](int64[:], int64[:], int64))

    @staticmethod
    def implementation(nonzero_degrees, nonzero_coeffs, primitive_element):  # pragma: no cover
        N = nonzero_degrees.size
        lambda_vector = nonzero_coeffs.copy()
        alpha_vector = np.zeros(N, dtype=nonzero_coeffs.dtype)
        for i in range(N):
            alpha_vector[i] = POWER(primitive_element, nonzero_degrees[i])
        degree = np.max(nonzero_degrees)
        roots = []
        powers = []

        # Test if 0 is a root
        if nonzero_degrees[-1] != 0:
            roots.append(0)
            powers.append(nonzero_degrees[-1])  # 0 has multiplicity equal to the lowest degree of x that is non-zero

        # Test if 1 is a root
        _sum = 0
        for i in range(N):
            _sum = ADD(_sum, lambda_vector[i])
        if _sum == 0:
            roots.append(1)
            powers.append(0)

        # Test if the powers of alpha are roots
        for i in range(1, ORDER - 1):
            _sum = 0
            for j in range(N):
                lambda_vector[j] = MULTIPLY(lambda_vector[j], alpha_vector[j])
                _sum = ADD(_sum, lambda_vector[j])
            if _sum == 0:
                root = POWER(primitive_element, i)
                roots.append(root)
                powers.append(i)
            if len(roots) == degree:
                break

        return np.array([roots, powers])
