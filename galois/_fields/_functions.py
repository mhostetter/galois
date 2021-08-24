"""
A module that contains a metaclass mixin that provides NumPy function overriding for an ndarray subclass. Additionally, other
JIT functions are created for use in polynomials and error-correcting codes, such as _poly_evaluate() or _poly_divmod().
"""
import numba
from numba import int64
import numpy as np

from . import _linalg
from ._dtypes import DTYPES
from ._ufuncs import UfuncMeta


class FunctionMeta(UfuncMeta):
    """
    A mixin metaclass that JIT compiles general-purpose functions on Galois field arrays.
    """
    # pylint: disable=no-value-for-parameter,abstract-method

    _UNSUPPORTED_FUNCTIONS_UNARY = [
        np.packbits, np.unpackbits,
        np.unwrap,
        np.around, np.round_, np.fix,
        np.gradient, np.trapz,
        np.i0, np.sinc,
        np.angle, np.real, np.imag, np.conj, np.conjugate,
    ]

    _UNSUPPORTED_FUNCTIONS_BINARY = [
        np.lib.scimath.logn,
        np.cross,
    ]

    _UNSUPPORTED_FUNCTIONS = _UNSUPPORTED_FUNCTIONS_UNARY + _UNSUPPORTED_FUNCTIONS_BINARY

    _FUNCTIONS_REQUIRING_VIEW = [
        np.concatenate,
        np.broadcast_to,
        np.trace,
    ]

    _OVERRIDDEN_FUNCTIONS = {
        np.convolve: "_convolve",
    }

    _OVERRIDDEN_LINALG_FUNCTIONS = {
        np.dot: _linalg.dot,
        np.vdot: _linalg.vdot,
        np.inner: _linalg.inner,
        np.outer: _linalg.outer,
        # np.tensordot: _linalg."tensordot",
        np.linalg.det: _linalg.det,
        np.linalg.matrix_rank: _linalg.matrix_rank,
        np.linalg.solve: _linalg.solve,
        np.linalg.inv: _linalg.inv,
    }

    _MATMUL_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], UfuncMeta._BINARY_CALCULATE_SIG, UfuncMeta._BINARY_CALCULATE_SIG, int64, int64, int64))
    _CONVOLVE_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], UfuncMeta._BINARY_CALCULATE_SIG, UfuncMeta._BINARY_CALCULATE_SIG, int64, int64, int64))
    _POLY_EVALUATE_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], UfuncMeta._BINARY_CALCULATE_SIG, UfuncMeta._BINARY_CALCULATE_SIG, int64, int64, int64))
    _POLY_DIVMOD_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:], UfuncMeta._BINARY_CALCULATE_SIG, UfuncMeta._BINARY_CALCULATE_SIG, UfuncMeta._BINARY_CALCULATE_SIG, int64, int64, int64))
    _POLY_ROOTS_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:], int64[:], int64, UfuncMeta._BINARY_CALCULATE_SIG, UfuncMeta._BINARY_CALCULATE_SIG, UfuncMeta._BINARY_CALCULATE_SIG, int64, int64, int64))

    _FUNCTION_CACHE_CALCULATE = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._functions = {}

    ###############################################################################
    # Individual functions, pre-compiled (cached)
    ###############################################################################

    def _function(cls, name):
        """
        Returns the function for the specific routine. The function compilation is based on `ufunc_mode`.
        """
        if name not in cls._functions:
            if cls.ufunc_mode != "python-calculate":
                cls._functions[name] = cls._function_calculate(name)
            else:
                cls._functions[name] = cls._function_python(name)
        return cls._functions[name]

    def _function_calculate(cls, name):
        """
        Returns a JIT-compiled function using explicit calculation. These functions are once-compiled and shared for all
        Galois fields. The only difference between Galois fields are the arithmetic funcs, characteristic, degree, and
        irreducible polynomial that are passed in as inputs.
        """
        key = (name,)

        if key not in cls._FUNCTION_CACHE_CALCULATE:
            function = getattr(cls, f"_{name}_calculate")
            sig = getattr(cls, f"_{name.upper()}_CALCULATE_SIG")
            cls._FUNCTION_CACHE_CALCULATE[key] = numba.jit(sig.signature, nopython=True, cache=True)(function)

        return cls._FUNCTION_CACHE_CALCULATE[key]

    def _function_python(cls, name):
        """
        Returns a pure-python function using explicit calculation.
        """
        return getattr(cls, f"_{name}_calculate")

    ###############################################################################
    # Function routines
    ###############################################################################

    def _matmul(cls, A, B, out=None, **kwargs):  # pylint: disable=unused-argument
        if not type(A) is type(B):
            raise TypeError(f"Operation 'matmul' requires both arrays be in the same Galois field, not {type(A)} and {type(B)}.")
        if not (A.ndim >= 1 and B.ndim >= 1):
            raise ValueError(f"Operation 'matmul' requires both arrays have dimension at least 1, not {A.ndim}-D and {B.ndim}-D.")
        if not (A.ndim <= 2 and B.ndim <= 2):
            raise ValueError("Operation 'matmul' currently only supports matrix multiplication up to 2-D. If you would like matrix multiplication of N-D arrays, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")
        field = type(A)
        dtype = A.dtype

        if field.is_prime_field:
            return _linalg._lapack_linalg(A, B, np.matmul, out=out)

        prepend, append = False, False
        if A.ndim == 1:
            A = A.reshape((1,A.size))
            prepend = True
        if B.ndim == 1:
            B = B.reshape((B.size,1))
            append = True

        if not A.shape[-1] == B.shape[-2]:
            raise ValueError(f"Operation 'matmul' requires the last dimension of A to match the second-to-last dimension of B, not {A.shape} and {B.shape}.")

        # if A.ndim > 2 and B.ndim == 2:
        #     new_shape = list(A.shape[:-2]) + list(B.shape)
        #     B = np.broadcast_to(B, new_shape)
        # if B.ndim > 2 and A.ndim == 2:
        #     new_shape = list(B.shape[:-2]) + list(A.shape)
        #     A = np.broadcast_to(A, new_shape)

        if cls.ufunc_mode != "python-calculate":
            A = A.astype(np.int64)
            B = B.astype(np.int64)
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            C = cls._function("matmul")(A, B, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
            C = C.astype(dtype)
        else:
            A = A.view(np.ndarray)
            B = B.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            C = cls._function("matmul")(A, B, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
        C = C.view(field)

        shape = list(C.shape)
        if prepend:
            shape = shape[1:]
        if append:
            shape = shape[:-1]
        C = C.reshape(shape)

        # TODO: Determine a better way to do this
        if out is not None:
            assert isinstance(out, tuple) and len(out) == 1  # TODO: Why is `out` getting populated as tuple?
            out = out[0]
            out[:] = C[:]

        return C

    def _convolve(cls, a, b, mode="full"):
        if not type(a) is type(b):
            raise TypeError(f"Arguments `a` and `b` must be of the same Galois field array class, not {type(a)} and {type(b)}.")
        if not mode == "full":
            raise ValueError(f"Operation 'convolve' currently only supports mode of 'full', not {mode!r}.")
        field = type(a)
        dtype = a.dtype

        if field.is_prime_field:
            # Determine the minimum dtype to hold the entire product and summation without overflowing
            n_sum = min(a.size, b.size)
            max_value = n_sum * (field.characteristic - 1)**2
            dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= max_value]
            dtype = np.object_ if len(dtypes) == 0 else dtypes[0]
            return_dtype = a.dtype
            a = a.view(np.ndarray).astype(dtype)
            b = b.view(np.ndarray).astype(dtype)
            c = np.convolve(a, b)  # Compute result using native numpy LAPACK/BLAS implementation
            c = c % field.characteristic  # Reduce the result mod p
            c = c.astype(return_dtype).view(field) if not np.isscalar(c) else field(c, dtype=return_dtype)
            return c
        else:
            if cls.ufunc_mode != "python-calculate":
                a = a.astype(np.int64)
                b = b.astype(np.int64)
                add = cls._func_calculate("add")
                multiply = cls._func_calculate("multiply")
                c = cls._function("convolve")(a, b, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
                c = c.astype(dtype)
            else:
                a = a.view(np.ndarray)
                b = b.view(np.ndarray)
                add = cls._func_python("add")
                multiply = cls._func_python("multiply")
                c = cls._function("convolve")(a, b, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
            c = c.view(field)

            return c

    def _poly_evaluate(cls, coeffs, x):
        field = cls
        dtype = x.dtype
        shape = x.shape
        x = np.atleast_1d(x.flatten())

        if cls.ufunc_mode != "python-calculate":
            coeffs = coeffs.astype(np.int64)
            x = x.astype(np.int64)
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            results = cls._function("poly_evaluate")(coeffs, x, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
            results = results.astype(dtype)
        else:
            coeffs = coeffs.view(np.ndarray)
            x = x.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            results = cls._function("poly_evaluate")(coeffs, x, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
        results = results.view(field)
        results = results.reshape(shape)

        return results

    def _poly_divmod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert 1 <= a.ndim <= 2 and b.ndim == 1
        field = type(a)
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
            qr = cls._function("poly_divmod")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, cls._irreducible_poly_int)
            qr = qr.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            qr = cls._function("poly_divmod")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, cls._irreducible_poly_int)
        qr = qr.view(field)

        q = qr[:, 0:q_degree + 1]
        r = qr[:, q_degree + 1:q_degree + 1 + r_degree + 1]

        if a_1d:
            q = q.reshape(q.size)
            r = r.reshape(r.size)

        return q, r

    def _poly_roots(cls, nonzero_degrees, nonzero_coeffs):
        assert isinstance(nonzero_coeffs, cls)
        field = cls
        dtype = nonzero_coeffs.dtype

        if cls.ufunc_mode != "python-calculate":
            nonzero_degrees = nonzero_degrees.astype(np.int64)
            nonzero_coeffs = nonzero_coeffs.astype(np.int64)
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            power = cls._func_calculate("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, np.int64(cls.primitive_element), add, multiply, power, cls.characteristic, cls.degree, cls._irreducible_poly_int)[0,:]
            roots = roots.astype(dtype)
        else:
            nonzero_degrees = nonzero_degrees.view(np.ndarray)
            nonzero_coeffs = nonzero_coeffs.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            power = cls._func_python("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, int(cls.primitive_element), add, multiply, power, cls.characteristic, cls.degree, cls._irreducible_poly_int)[0,:]
        roots = roots.view(field)

        idxs = np.argsort(roots)
        return roots[idxs]

    ###############################################################################
    # Function implementations using explicit calculation
    ###############################################################################

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _matmul_calculate(A, B, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = A.dtype

        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[-1] == B.shape[-2]

        M, K = A.shape
        K, N = B.shape
        C = np.zeros((M, N), dtype=dtype)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    C[i,j] = ADD(C[i,j], MULTIPLY(A[i,k], B[k,j], *args), *args)

        return C

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _convolve_calculate(a, b, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        c = np.zeros(a.size + b.size - 1, dtype=dtype)
        for i in range(a.size):
            for j in range(b.size - 1, -1, -1):
                c[i + j] = ADD(c[i + j], MULTIPLY(a[i], b[j], *args), *args)

        return c

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _poly_evaluate_calculate(coeffs, values, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = values.dtype

        results = np.zeros(values.size, dtype=dtype)
        for i in range(values.size):
            results[i] = coeffs[0]
            for j in range(1, coeffs.size):
                results[i] = ADD(coeffs[j], MULTIPLY(results[i], values[i], *args), *args)

        return results

    @staticmethod
    @numba.extending.register_jitable(inline="always")
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
                    for j in range(b.size):
                        qr[k, i + j] = SUBTRACT(qr[k, i + j], MULTIPLY(q, b[j], *args), *args)
                    qr[k,i] = q

        return qr

    @staticmethod
    @numba.extending.register_jitable(inline="always")
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
