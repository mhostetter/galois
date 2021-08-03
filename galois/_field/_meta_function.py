"""
A module containint a metaclass mixin abstract base class that handles overridding numpy function calls.
"""
import numba
from numba import int64
import numpy as np

from ._dtypes import DTYPES

from . import _linalg
from ._meta_properties import PropertiesMeta

# List of functions that are not valid on arrays over finite groups, rings, and fields
UNSUPPORTED_FUNCTIONS_UNARY = [
    np.packbits, np.unpackbits,
    np.unwrap,
    np.around, np.round_, np.fix,
    np.gradient, np.trapz,
    np.i0, np.sinc,
    np.angle, np.real, np.imag, np.conj, np.conjugate,
]

UNSUPPORTED_FUNCTIONS_BINARY = [
    np.lib.scimath.logn,
    np.cross,
]

FUNCTIONS_REQUIRING_VIEW = [
    np.concatenate,
    np.broadcast_to,
    np.trace,
]


class FunctionMeta(PropertiesMeta):
    """
    A mixin metaclass that JIT compiles general-purpose functions on Galois field arrays.
    """
    # pylint: disable=no-value-for-parameter

    _unsupported_functions = UNSUPPORTED_FUNCTIONS_UNARY + UNSUPPORTED_FUNCTIONS_BINARY
    _functions_requiring_view = FUNCTIONS_REQUIRING_VIEW

    _overridden_functions = {
        np.convolve: "_convolve",
    }

    _overridden_linalg_functions = {
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

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._gufuncs = {}
        cls._functions = {}

    ###############################################################################
    # Individual gufuncs, compiled on-demand
    ###############################################################################

    def _gufunc(cls, name):
        if cls._gufuncs.get(name, None) is None:
            if name == "poly_evaluate":
                if cls.ufunc_mode == "python-calculate":
                    cls._gufuncs["poly_evaluate"] = np.vectorize(poly_evaluate_python_calculate, excluded=["coeffs"], otypes=[np.object_])
                else:
                    global ADD_JIT, MULTIPLY_JIT
                    ADD_JIT = cls._calculate_jit("add")
                    MULTIPLY_JIT = cls._calculate_jit("multiply")
                    cls._gufuncs["poly_evaluate"] = numba.guvectorize("int64[:], int64[:], int64, int64, int64, int64[:]", "(m),(n),(),(),()->(n)", nopython=True)(poly_evaluate_gufunc_calculate)
            else:
                raise NotImplementedError
        return cls._gufuncs[name]

    ###############################################################################
    # Individual functions, pre-compiled (cached)
    ###############################################################################

    def _function(cls, name):
        if cls._functions.get(name, None) is None:
            if cls.ufunc_mode != "python-calculate":
                cls._functions[name] = jit_calculate(name)
            else:
                cls._functions[name] = python_func(name)
        return cls._functions[name]

    ###############################################################################
    # Gufunc routines
    ###############################################################################

    def _poly_evaluate(cls, coeffs, x):
        assert isinstance(coeffs, cls) and isinstance(x, cls)
        assert coeffs.ndim == 1
        field = cls
        dtype = x.dtype
        x = np.atleast_1d(x)

        if cls.ufunc_mode == "python-calculate":
            # For object dtypes, call the vectorized classmethod
            add = cls._python_func("add")
            multiply = cls._python_func("multiply")
            y = cls._gufunc("poly_evaluate")(coeffs=coeffs.view(np.ndarray), value=x.view(np.ndarray), ADD=add, MULTIPLY=multiply, CHARACTERISTIC=cls.characteristic, DEGREE=cls.degree, IRREDUCIBLE_POLY=cls._irreducible_poly_int, result=field.Zeros(x.shape))
        else:
            # For integer dtypes, call the JIT-compiled gufunc
            y = cls._gufunc("poly_evaluate")(coeffs, x, cls.characteristic, cls.degree, cls._irreducible_poly_int, field.Zeros(x.shape), casting="unsafe")
            y = y.astype(dtype)
        y = y.view(field)

        if y.size == 1:
            y = y[0]

        return y

    def _poly_evaluate_python(cls, coeffs, x):
        assert coeffs.ndim == 1
        add = cls._python_func("add")
        multiply = cls._python_func("multiply")
        coeffs = coeffs.view(np.ndarray).astype(cls.dtypes[-1])
        x = int(x)
        y = poly_evaluate_python_calculate(coeffs, x, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int, 0)
        y = cls(y)
        return y

    ###############################################################################
    # JIT function routines
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
            add = cls._calculate_jit("add")
            multiply = cls._calculate_jit("multiply")
            C = cls._function("matmul")(A, B, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
            C = C.astype(dtype)
        else:
            A = A.view(np.ndarray)
            B = B.view(np.ndarray)
            add = cls._python_func("add")
            multiply = cls._python_func("multiply")
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
            if cls._ufunc_mode != "python-calculate":
                a = a.astype(np.int64)
                b = b.astype(np.int64)
                add = cls._calculate_jit("add")
                multiply = cls._calculate_jit("multiply")
                c = cls._function("convolve")(a, b, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
                c = c.astype(dtype)
            else:
                a = a.view(np.ndarray)
                b = b.view(np.ndarray)
                add = cls._python_func("add")
                multiply = cls._python_func("multiply")
                c = cls._function("convolve")(a, b, add, multiply, cls.characteristic, cls.degree, cls._irreducible_poly_int)
            c = c.view(field)

            return c

    def _poly_divmod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert 1 <= a.ndim <= 2 and b.ndim == 1
        field = type(a)
        dtype = a.dtype
        a_1d = a.ndim == 1
        a = np.atleast_2d(a)

        q_degree = a.shape[-1] - b.shape[-1]
        r_degree = b.shape[-1] - 1

        if cls._ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            subtract = cls._calculate_jit("subtract")
            multiply = cls._calculate_jit("multiply")
            divide = cls._calculate_jit("divide")
            qr = cls._function("poly_divmod")(a, b, subtract, multiply, divide, cls.characteristic, cls.degree, cls._irreducible_poly_int)
            qr = qr.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._python_func("subtract")
            multiply = cls._python_func("multiply")
            divide = cls._python_func("divide")
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

        if cls._ufunc_mode != "python-calculate":
            nonzero_degrees = nonzero_degrees.astype(np.int64)
            nonzero_coeffs = nonzero_coeffs.astype(np.int64)
            add = cls._calculate_jit("add")
            multiply = cls._calculate_jit("multiply")
            power = cls._calculate_jit("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, np.int64(cls.primitive_element), add, multiply, power, cls.characteristic, cls.degree, cls._irreducible_poly_int)[0,:]
            roots = roots.astype(dtype)
        else:
            nonzero_degrees = nonzero_degrees.view(np.ndarray)
            nonzero_coeffs = nonzero_coeffs.view(np.ndarray)
            add = cls._python_func("add")
            multiply = cls._python_func("multiply")
            power = cls._python_func("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, int(cls.primitive_element), add, multiply, power, cls.characteristic, cls.degree, cls._irreducible_poly_int)[0,:]
        roots = roots.view(field)

        idxs = np.argsort(roots)
        return roots[idxs]


##############################################################################
# Individual gufuncs
##############################################################################

ADD_JIT = lambda a, b, *args: a + b
MULTIPLY_JIT = lambda a, b, *args: a * b

# pylint: disable=redefined-outer-name,unused-argument


def poly_evaluate_gufunc_lookup(coeffs, values, ADD, MULTIPLY, EXP, LOG, ZECH_LOG, ZECH_E, results):  # pragma: no cover
    args = EXP, LOG, ZECH_LOG, ZECH_E

    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD(coeffs[j], MULTIPLY(results[i], values[i], *args), *args)


def poly_evaluate_gufunc_calculate(coeffs, values, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY, results):  # pragma: no cover
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY

    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i], *args), *args)


def poly_evaluate_python_calculate(coeffs, value, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY, result):  # pragma: no cover
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY

    result = coeffs[0]
    for j in range(1, coeffs.size):
        result = ADD(coeffs[j], MULTIPLY(result, value, *args), *args)

    return result


###############################################################################
# Compilation functions
###############################################################################

UNARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64[:], int64[:], int64[:], int64))
BINARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64, int64[:], int64[:], int64[:], int64))
UNARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64))
BINARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64, int64))


def jit_lookup(name):
    if name not in jit_lookup.cache:
        function = eval(f"{name}_lookup")
        sig = eval(f"{name.upper()}_LOOKUP_SIG")
        jit_calculate.cache[name] = numba.jit(sig.signature, nopython=True, cache=True)(function)
    return jit_lookup.cache[name]

jit_lookup.cache = {}


def jit_calculate(name):
    if name not in jit_calculate.cache:
        function = eval(f"{name}_calculate")
        sig = eval(f"{name.upper()}_CALCULATE_SIG")
        jit_calculate.cache[name] = numba.jit(sig.signature, nopython=True, cache=True)(function)
    return jit_calculate.cache[name]

jit_calculate.cache = {}


def python_func(name):
    return eval(f"{name}_calculate")


###############################################################################
# Matrix multiplication
###############################################################################

MATMUL_LOOKUP_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], BINARY_LOOKUP_SIG, BINARY_LOOKUP_SIG, int64[:], int64[:], int64[:], int64))

def matmul_lookup(A, B, ADD, MULTIPLY, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    args = EXP, LOG, ZECH_LOG, ZECH_E
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


MATMUL_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, int64, int64, int64))

def matmul_calculate(A, B, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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


###############################################################################
# Convolution
###############################################################################

CONVOLVE_LOOKUP_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], BINARY_LOOKUP_SIG, BINARY_LOOKUP_SIG, int64[:], int64[:], int64[:], int64))

def convolve_lookup(a, b, ADD, MULTIPLY, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    args = EXP, LOG, ZECH_LOG, ZECH_E
    dtype = a.dtype

    c = np.zeros(a.size + b.size - 1, dtype=dtype)
    for i in range(a.size):
        for j in range(b.size - 1, -1, -1):
            c[i + j] = ADD(c[i + j], MULTIPLY(a[i], b[j], *args), *args)

    return c


CONVOLVE_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, int64, int64, int64))

def convolve_calculate(a, b, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
    dtype = a.dtype

    c = np.zeros(a.size + b.size - 1, dtype=dtype)
    for i in range(a.size):
        for j in range(b.size - 1, -1, -1):
            c[i + j] = ADD(c[i + j], MULTIPLY(a[i], b[j], *args), *args)

    return c


###############################################################################
# Polynomial division with remainder
###############################################################################

POLY_DIVMOD_LOOKUP_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:], BINARY_LOOKUP_SIG, BINARY_LOOKUP_SIG, BINARY_LOOKUP_SIG, int64[:], int64[:], int64[:], int64))

def poly_divmod_lookup(a, b, SUBTRACT, MULTIPLY, DIVIDE, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    args = EXP, LOG, ZECH_LOG, ZECH_E

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


POLY_DIVMOD_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:], BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, int64, int64, int64))

def poly_divmod_calculate(a, b, SUBTRACT, MULTIPLY, DIVIDE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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


###############################################################################
# Polynomial roots
###############################################################################

POLY_ROOTS_LOOKUP_SIG = numba.types.FunctionType(int64[:,:](int64[:], int64[:], int64, BINARY_LOOKUP_SIG, BINARY_LOOKUP_SIG, BINARY_LOOKUP_SIG, int64[:], int64[:], int64[:], int64))

def poly_roots_lookup(nonzero_degrees, nonzero_coeffs, primitive_element, ADD, MULTIPLY, POWER, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    args = EXP, LOG, ZECH_LOG, ZECH_E
    dtype = nonzero_coeffs.dtype
    ORDER = LOG.size

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


POLY_ROOTS_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:], int64[:], int64, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, int64, int64, int64))

def poly_roots_calculate(nonzero_degrees, nonzero_coeffs, primitive_element, ADD, MULTIPLY, POWER, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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


###############################################################################
# Polynomial evaluation
###############################################################################

POLY_EVALUATE_LOOKUP_SIG = numba.types.FunctionType(int64(int64[:], int64, BINARY_LOOKUP_SIG, BINARY_LOOKUP_SIG, int64[:], int64[:], int64[:], int64))

def poly_evaluate_lookup(coeffs, value, ADD, MULTIPLY, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    args = EXP, LOG, ZECH_LOG, ZECH_E

    result = coeffs[0]
    for j in range(1, coeffs.size):
        result = ADD(coeffs[j], MULTIPLY(result, value, *args), *args)

    return result


POLY_EVALUATE_CALCULATE_SIG = numba.types.FunctionType(int64(int64[:], int64, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, int64, int64, int64))

def poly_evaluate_calculate(coeffs, value, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY

    result = coeffs[0]
    for j in range(1, coeffs.size):
        result = ADD(coeffs[j], MULTIPLY(result, value, *args), *args)

    return result


# Set globals so that when the application-specific python functions are called they will use the
# pure-python arithmetic routines
MATMUL = matmul_calculate
CONVOLVE = convolve_calculate
POLY_DIVMOD = poly_divmod_calculate
POLY_ROOTS = poly_roots_calculate
