"""
A module that contains Array mixin classes that implement JIT-compiled polynomial arithmetic.
"""
import abc

import numba
from numba import int64, uint64
import numpy as np

from ._function import RingFunctions, FieldFunctions

ORDER: int
PRIMITIVE_ELEMENT: int

ADD = np.add
SUBTRACT = np.subtract
MULTIPLY = np.multiply
DIVIDE = np.divide
POWER = np.power

POLY_MULTIPLY = np.convolve
POLY_MOD = lambda x, y: x % y


class RingPolyFunctions(RingFunctions, abc.ABC):
    """
    A mixin base class that ring arithmetic (+, -, *) on polynomials, using *only* explicit calculation.
    """

    ###############################################################################
    # Polynomial evaluation
    ###############################################################################

    @classmethod
    def _poly_evaluate(cls, coeffs, x):
        dtype = x.dtype
        shape = x.shape
        x = np.atleast_1d(x.flatten())

        if cls.ufunc_mode != "python-calculate":
            coeffs = coeffs.astype(np.int64)
            x = x.astype(np.int64)
            results = cls._function("poly_evaluate")(coeffs, x)
            results = results.astype(dtype)
        else:
            coeffs = coeffs.view(np.ndarray)
            x = x.view(np.ndarray)
            results = cls._function("poly_evaluate")(coeffs, x)
        results = cls._view(results)
        results = results.reshape(shape)

        return results

    @classmethod
    def _poly_evaluate_matrix(cls, coeffs, X):
        assert X.ndim == 2 and X.shape[0] == X.shape[1]
        I = cls.Identity(X.shape[0])

        results = coeffs[0]*I
        for j in range(1, coeffs.size):
            results = coeffs[j]*I + results @ X

        return results

    @classmethod
    def _set_poly_evaluate_jit_globals(cls):
        global ADD, MULTIPLY
        ADD = cls._ufunc("add")
        MULTIPLY = cls._ufunc("multiply")

    _POLY_EVALUATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_evaluate_jit(coeffs, values):
        results = np.zeros(values.size, dtype=values.dtype)
        for i in range(values.size):
            results[i] = coeffs[0]
            for j in range(1, coeffs.size):
                # NOTE: This workaround, using results[i:i+1] instead of results[i], is needed because if ADD = np.bitwise_xor for
                # GF(2^m) will return np.intc instead of np.object_ with int. However, if the input to the ufunc is a dtype=object
                # array, then the type is preserved. There may be a better way to do this.
                results[i:i+1] = ADD(coeffs[j], MULTIPLY(results[i:i+1], values[i]))

        return results

    ###############################################################################
    # Polynomial roots
    ###############################################################################

    @classmethod
    def _poly_roots(cls, nonzero_degrees, nonzero_coeffs):
        assert isinstance(nonzero_coeffs, cls)
        dtype = nonzero_coeffs.dtype

        if cls.ufunc_mode != "python-calculate":
            nonzero_degrees = nonzero_degrees.astype(np.int64)
            nonzero_coeffs = nonzero_coeffs.astype(np.int64)
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, int(cls.primitive_element))[0,:]
            roots = roots.astype(dtype)
        else:
            nonzero_degrees = nonzero_degrees.view(np.ndarray)
            nonzero_coeffs = nonzero_coeffs.view(np.ndarray)
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, int(cls.primitive_element))[0,:]
        roots = cls._view(roots)

        idxs = np.argsort(roots)
        return roots[idxs]

    @classmethod
    def _set_poly_roots_jit_globals(cls):
        global ORDER, ADD, MULTIPLY, POWER
        ORDER = cls.order
        ADD = cls._ufunc("add")
        MULTIPLY = cls._ufunc("multiply")
        POWER = cls._ufunc("power")

    _POLY_ROOTS_SIG = numba.types.FunctionType(int64[:,:](int64[:], int64[:], int64))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_roots_jit(nonzero_degrees, nonzero_coeffs, primitive_element):
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
            powers.append(-1)

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

        return np.array([roots, powers], dtype=nonzero_coeffs.dtype)


class FieldPolyFunctions(RingPolyFunctions, FieldFunctions, abc.ABC):
    """
    A mixin base class that field arithmetic (+, -, *, /) on polynomials, using *only* explicit calculation.
    """

    ###############################################################################
    # Polynomial floor division
    ###############################################################################

    @classmethod
    def _poly_floordiv(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if cls.ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            q = cls._function("poly_floordiv")(a, b)
            q = q.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            q = cls._function("poly_floordiv")(a, b)
        q = cls._view(q)

        return q

    @classmethod
    def _set_poly_floordiv_jit_globals(cls):
        global SUBTRACT, MULTIPLY, DIVIDE
        SUBTRACT = cls._ufunc("subtract")
        MULTIPLY = cls._ufunc("multiply")
        DIVIDE = cls._ufunc("divide")

    _POLY_FLOORDIV_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_floordiv_jit(a, b):
        if b.size == 1 and b[0] == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")

        if a.size < b.size:
            return np.array([0], dtype=a.dtype)

        q_degree = a.size - b.size
        q = np.zeros(q_degree + 1, dtype=a.dtype)
        aa = a[0:q_degree + 1].copy()

        for i in range(q_degree + 1):
            if aa[i] > 0:
                q[i] = DIVIDE(aa[i], b[0])
                N = min(b.size, q_degree + 1 - i)  # We don't need to subtract in the "remainder" range
                for j in range(1, N):
                    aa[i + j] = SUBTRACT(aa[i + j], MULTIPLY(q[i], b[j]))

        return q

    ###############################################################################
    # Polynomial modular division
    ###############################################################################

    @classmethod
    def _poly_mod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if cls.ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            r = cls._function("poly_mod")(a, b)
            r = r.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            r = cls._function("poly_mod")(a, b)
        r = cls._view(r)

        return r

    @classmethod
    def _set_poly_mod_jit_globals(cls):
        global SUBTRACT, MULTIPLY, DIVIDE
        SUBTRACT = cls._ufunc("subtract")
        MULTIPLY = cls._ufunc("multiply")
        DIVIDE = cls._ufunc("divide")

    _POLY_MOD_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_mod_jit(a, b):
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
                q = DIVIDE(r[0], b[0])
                for j in range(1, b.size):
                    r[j] = SUBTRACT(r[j], MULTIPLY(q, b[j]))

        r = r[1:]

        # Trim leading zeros to reduce computations in future calls
        if r.size > 1:
            idxs = np.nonzero(r)[0]
            if idxs.size > 0:
                r = r[idxs[0]:]
            else:
                r = r[-1:]

        return r

    ###############################################################################
    # Polynomial division with remainder
    ###############################################################################

    @classmethod
    def _poly_divmod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert 1 <= a.ndim <= 2 and b.ndim == 1
        dtype = a.dtype
        a_1d = a.ndim == 1
        a = np.atleast_2d(a)

        q_degree = a.shape[-1] - b.shape[-1]
        r_degree = b.shape[-1] - 1

        if cls.ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            qr = cls._function("poly_divmod")(a, b)
            qr = qr.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            qr = cls._function("poly_divmod")(a, b)
        qr = cls._view(qr)

        q = qr[:, 0:q_degree + 1]
        r = qr[:, q_degree + 1:q_degree + 1 + r_degree + 1]

        if a_1d:
            q = q.reshape(q.size)
            r = r.reshape(r.size)

        return q, r

    @classmethod
    def _set_poly_divmod_jit_globals(cls):
        global SUBTRACT, MULTIPLY, DIVIDE
        SUBTRACT = cls._ufunc("subtract")
        MULTIPLY = cls._ufunc("multiply")
        DIVIDE = cls._ufunc("divide")

    _POLY_DIVMOD_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:]))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_divmod_jit(a, b):
        assert a.ndim == 2 and b.ndim == 1
        assert a.shape[-1] >= b.shape[-1]

        q_degree = a.shape[1] - b.shape[-1]
        qr = a.copy()

        for k in range(a.shape[0]):
            for i in range(q_degree + 1):
                if qr[k,i] > 0:
                    q = DIVIDE(qr[k,i], b[0])
                    for j in range(1, b.size):
                        qr[k, i + j] = SUBTRACT(qr[k, i + j], MULTIPLY(q, b[j]))
                    qr[k,i] = q

        return qr

    ###############################################################################
    # Polynomial modular exponentiation
    ###############################################################################

    # TODO: Make this work with rings (without modular reduction)

    @classmethod
    def _poly_pow(cls, a, b, c=None):
        assert isinstance(a, cls) and isinstance(b, (int, np.integer)) and isinstance(c, (type(None), cls))
        assert a.ndim == 1 and c.ndim == 1 if c is not None else True
        dtype = a.dtype

        # Convert the integer b into a vector of uint64 [MSWord, ..., LSWord] so arbitrarily-large exponents may be
        # passed into the JIT-compiled version
        b_vec = []  # Pop on LSWord -> MSWord
        while b > 2**64:
            q, r = divmod(b, 2**64)
            b_vec.append(r)
            b = q
        b_vec.append(b)
        b_vec = np.array(b_vec[::-1], dtype=np.uint64)  # Make vector MSWord -> LSWord

        if cls.ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            c = np.array([], dtype=np.int64) if c is None else c.astype(np.int64)
            z = cls._function("poly_pow")(a, b_vec, c)
            z = z.astype(dtype)
        else:
            a = a.view(np.ndarray)
            c = np.array([], dtype=dtype) if c is None else c.view(np.ndarray)
            z = cls._function("poly_pow")(a, b_vec, c)
        z = cls._view(z)

        return z

    @classmethod
    def _set_poly_pow_jit_globals(cls):
        global POLY_MULTIPLY, POLY_MOD
        POLY_MULTIPLY = cls._function("convolve")
        POLY_MOD = cls._function("poly_mod")

    _POLY_POW_SIG = numba.types.FunctionType(int64[:](int64[:], uint64[:], int64[:]))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_pow_jit(a, b_vec, c):
        """
        b is a vector of uint64 [MSWord, ..., LSWord] so that arbitrarily-large exponents may be passed
        """
        if b_vec.size == 1 and b_vec[0] == 0:
            return np.array([1], dtype=a.dtype)

        result_s = a.copy()  # The "squaring" part
        result_m = np.array([1], dtype=a.dtype)  # The "multiplicative" part

        # Loop from LSWord to MSWord
        for i in range(b_vec.size - 1, -1, -1):
            j = 0  # Bit counter -- make sure we interate through 64 bits on all but the most-significant word
            while j < 64:
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
