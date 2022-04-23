"""
A module that contains Array mixin classes that implement JIT-compiled polynomial arithmetic.
"""
import abc

import numba
from numba import int64, uint64
import numpy as np

from ._function import RingFunction, FieldFunction


class RingPolyFunction(RingFunction, abc.ABC):
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
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            results = cls._function("poly_evaluate")(coeffs, x, add, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            results = results.astype(dtype)
        else:
            coeffs = coeffs.view(np.ndarray)
            x = x.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            results = cls._function("poly_evaluate")(coeffs, x, add, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
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

    _POLY_EVALUATE_CALCULATE_SIG = numba.types.FunctionType(int64[:](
        int64[:],
        int64[:],
        RingFunction._BINARY_CALCULATE_SIG,
        RingFunction._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_evaluate_calculate(coeffs, values, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = values.dtype

        results = np.zeros(values.size, dtype=dtype)
        for i in range(values.size):
            results[i] = coeffs[0]
            for j in range(1, coeffs.size):
                results[i] = ADD(coeffs[j], MULTIPLY(results[i], values[i], *args), *args)

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
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            power = cls._func_calculate("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, np.int64(cls.primitive_element), add, multiply, power, cls.characteristic, cls.degree, int(cls.irreducible_poly))[0,:]
            roots = roots.astype(dtype)
        else:
            nonzero_degrees = nonzero_degrees.view(np.ndarray)
            nonzero_coeffs = nonzero_coeffs.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            power = cls._func_python("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, int(cls.primitive_element), add, multiply, power, cls.characteristic, cls.degree, int(cls.irreducible_poly))[0,:]
        roots = cls._view(roots)

        idxs = np.argsort(roots)
        return roots[idxs]

    _POLY_ROOTS_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](
        int64[:],
        int64[:],
        int64,
        RingFunction._BINARY_CALCULATE_SIG,
        RingFunction._BINARY_CALCULATE_SIG,
        RingFunction._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_roots_calculate(nonzero_degrees, nonzero_coeffs, primitive_element, ADD, MULTIPLY, POWER, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = nonzero_coeffs.dtype
        ORDER = CHARACTERISTIC**DEGREE

        N = nonzero_degrees.size
        lambda_vector = nonzero_coeffs.copy()
        alpha_vector = np.zeros(N, dtype=dtype)
        for i in range(N):
            alpha_vector[i] = POWER(primitive_element, nonzero_degrees[i], *args)
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
            _sum = ADD(_sum, lambda_vector[i], *args)
        if _sum == 0:
            roots.append(1)
            powers.append(0)

        # Test if the powers of alpha are roots
        for i in range(1, ORDER - 1):
            _sum = 0
            for j in range(N):
                lambda_vector[j] = MULTIPLY(lambda_vector[j], alpha_vector[j], *args)
                _sum = ADD(_sum, lambda_vector[j], *args)
            if _sum == 0:
                root = POWER(primitive_element, i, *args)
                roots.append(root)
                powers.append(i)
            if len(roots) == degree:
                break

        return np.array([roots, powers], dtype=dtype)


class FieldPolyFunction(RingPolyFunction, FieldFunction, abc.ABC):
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
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            q = cls._function("poly_floordiv")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            q = q.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            q = cls._function("poly_floordiv")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, int(cls.irreducible_poly))
        q = cls._view(q)

        return q

    _POLY_FLOORDIV_CALCULATE_SIG = numba.types.FunctionType(int64[:](
        int64[:],
        int64[:],
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_floordiv_calculate(a, b, SUBTRACT, MULTIPLY, DIVIDE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        if b.size == 1 and b[0] == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")

        if a.size < b.size:
            return np.array([0], dtype=dtype)

        q_degree = a.size - b.size
        q = np.zeros(q_degree + 1, dtype=dtype)
        aa = a[0:q_degree + 1].copy()

        for i in range(q_degree + 1):
            if aa[i] > 0:
                q[i] = DIVIDE(aa[i], b[0], *args)
                N = min(b.size, q_degree + 1 - i)  # We don't need to subtract in the "remainder" range
                for j in range(1, N):
                    aa[i + j] = SUBTRACT(aa[i + j], MULTIPLY(q[i], b[j], *args), *args)

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
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            r = cls._function("poly_mod")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            r = r.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            r = cls._function("poly_mod")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, int(cls.irreducible_poly))
        r = cls._view(r)

        return r

    _POLY_MOD_CALCULATE_SIG = numba.types.FunctionType(int64[:](
        int64[:],
        int64[:],
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_mod_calculate(a, b, SUBTRACT, MULTIPLY, DIVIDE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        if b.size == 1 and b[0] == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")

        if a.size < b.size:
            return a.copy()

        if b.size == 1:
            return np.array([0], dtype=dtype)

        q_degree = a.size - b.size
        r_degree = b.size - 1
        r = np.zeros(r_degree + 1, dtype=dtype)
        r[1:] = a[0:r_degree]

        for i in range(q_degree + 1):
            r = np.roll(r, -1)
            r[-1] = a[i + r_degree]

            if r[0] > 0:
                q = DIVIDE(r[0], b[0], *args)
                for j in range(1, b.size):
                    r[j] = SUBTRACT(r[j], MULTIPLY(q, b[j], *args), *args)

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
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            qr = cls._function("poly_divmod")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            qr = qr.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            qr = cls._function("poly_divmod")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, int(cls.irreducible_poly))
        qr = cls._view(qr)

        q = qr[:, 0:q_degree + 1]
        r = qr[:, q_degree + 1:q_degree + 1 + r_degree + 1]

        if a_1d:
            q = q.reshape(q.size)
            r = r.reshape(r.size)

        return q, r

    _POLY_DIVMOD_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](
        int64[:,:],
        int64[:],
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_divmod_calculate(a, b, SUBTRACT, MULTIPLY, DIVIDE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY

        assert a.ndim == 2 and b.ndim == 1
        assert a.shape[-1] >= b.shape[-1]

        q_degree = a.shape[1] - b.shape[-1]
        qr = a.copy()

        for k in range(a.shape[0]):
            for i in range(q_degree + 1):
                if qr[k,i] > 0:
                    q = DIVIDE(qr[k,i], b[0], *args)
                    for j in range(1, b.size):
                        qr[k, i + j] = SUBTRACT(qr[k, i + j], MULTIPLY(q, b[j], *args), *args)
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
            add = cls._func_calculate("add")
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            convolve = cls._function("convolve")
            poly_mod = cls._function("poly_mod")
            z = cls._function("poly_pow")(a, b_vec, c, add, subtract, multiply, divide, convolve, poly_mod, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            z = z.astype(dtype)
        else:
            a = a.view(np.ndarray)
            c = np.array([], dtype=dtype) if c is None else c.view(np.ndarray)
            add = cls._func_python("add")
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            convolve = cls._function("convolve")
            poly_mod = cls._function("poly_mod")
            z = cls._function("poly_pow")(a, b_vec, c, add, subtract, multiply, divide, convolve, poly_mod, cls.characteristic, cls.degree, int(cls.irreducible_poly))
        z = cls._view(z)

        return z

    _POLY_POW_CALCULATE_SIG = numba.types.FunctionType(int64[:](
        int64[:],
        uint64[:],
        int64[:],
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._BINARY_CALCULATE_SIG,
        FieldFunction._CONVOLVE_CALCULATE_SIG,
        _POLY_MOD_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_pow_calculate(a, b_vec, c, ADD, SUBTRACT, MULTIPLY, DIVIDE, POLY_MULTIPLY, POLY_MOD, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        b is a vector of uint64 [MSWord, ..., LSWord] so that arbitrarily-large exponents may be passed
        """
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        if b_vec.size == 1 and b_vec[0] == 0:
            return np.array([1], dtype=dtype)

        result_s = a.copy()  # The "squaring" part
        result_m = np.array([1], dtype=dtype)  # The "multiplicative" part

        # Loop from LSWord to MSWord
        for i in range(b_vec.size - 1, -1, -1):
            j = 0  # Bit counter -- make sure we interate through 64 bits on all but the most-significant word
            while j < 64:
                if i == 0 and b_vec[i] <= 1:
                    # This is the MSB and we already accounted for the most-significant bit -- can exit now
                    break

                if b_vec[i] % 2 == 0:
                    result_s = POLY_MULTIPLY(result_s, result_s, ADD, MULTIPLY, *args)
                    if c.size > 0:
                        result_s = POLY_MOD(result_s, c, SUBTRACT, MULTIPLY, DIVIDE, *args)
                    b_vec[i] //= 2
                    j += 1
                else:
                    result_m = POLY_MULTIPLY(result_m, result_s, ADD, MULTIPLY, *args)
                    if c.size > 0:
                        result_m = POLY_MOD(result_m, c, SUBTRACT, MULTIPLY, DIVIDE, *args)
                    b_vec[i] -= 1

        result = POLY_MULTIPLY(result_s, result_m, ADD, MULTIPLY, *args)
        if c.size > 0:
            result = POLY_MOD(result, c, SUBTRACT, MULTIPLY, DIVIDE, *args)

        return result
