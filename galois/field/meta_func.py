import numba
import numpy as np

from ..dtypes import DTYPES
from ..meta_func import Func

from .linalg import _lapack_linalg, dot, vdot, inner, outer, matrix_rank, solve, inv, det

ORDER = None  # The field's order `p^m`
PRIMITIVE_ELEMENT = None  # The field's primitive element in integer form

# Placeholder functions to be replaced by JIT-compiled function
ADD_UFUNC = lambda x, y: x + y
SUBTRACT_UFUNC = lambda x, y: x - y
MULTIPLY_UFUNC = lambda x, y: x * y
RECIPROCAL_UFUNC = lambda x: 1 / x
DIVIDE_UFUNC = lambda x, y: x // y
POWER_UFUNC = lambda x, y: x ** y


class FieldFunc(Func):
    """
    A mixin class that JIT compiles general purpose functions for polynomial arithmetic and convolution.
    """
    # pylint: disable=no-value-for-parameter

    _overridden_functions = {
        np.convolve: "_convolve",
    }

    _overridden_linalg_functions = {
        np.dot: dot,
        np.vdot: vdot,
        np.inner: inner,
        np.outer: outer,
        # np.tensordot: "tensordot",
        np.linalg.det: det,
        np.linalg.matrix_rank: matrix_rank,
        np.linalg.solve: solve,
        np.linalg.inv: inv,
    }

    ###############################################################################
    # Individual functions, compiled on-demand
    ###############################################################################

    def _func_matmul(cls):
        if cls._funcs.get("matmul", None) is None:
            if cls.ufunc_mode == "python-calculate":
                cls._funcs["matmul"] = cls._matmul_python
            else:
                global ADD_UFUNC, MULTIPLY_UFUNC
                ADD_UFUNC = cls._ufunc_add()
                MULTIPLY_UFUNC = cls._ufunc_multiply()
                assert cls.ufunc_target == "cpu"
                cls._funcs["matmul"] = numba.jit("int64[:,:](int64[:,:], int64[:,:])", nopython=True)(_matmul_jit)
        return cls._funcs["matmul"]

    def _func_convolve(cls):
        if cls._funcs.get("convolve", None) is None:
            if cls.ufunc_mode == "python-calculate":
                cls._funcs["convolve"] = cls._convolve_python
            else:
                global ADD_UFUNC, MULTIPLY_UFUNC
                ADD_UFUNC = cls._ufunc_add()
                MULTIPLY_UFUNC = cls._ufunc_multiply()
                assert cls.ufunc_target == "cpu"
                cls._funcs["convolve"] = numba.jit("int64[:](int64[:], int64[:])", nopython=True)(_convolve_jit)
        return cls._funcs["convolve"]

    def _func_poly_divmod(cls):
        if cls._funcs.get("poly_divmod", None) is None:
            if cls.ufunc_mode == "python-calculate":
                cls._funcs["poly_divmod"] = cls._poly_divmod_python
            else:
                global SUBTRACT_UFUNC, MULTIPLY_UFUNC, DIVIDE_UFUNC
                SUBTRACT_UFUNC = cls._ufunc_subtract()
                MULTIPLY_UFUNC = cls._ufunc_multiply()
                DIVIDE_UFUNC = cls._ufunc_divide()
                assert cls.ufunc_target == "cpu"
                cls._funcs["poly_divmod"] = numba.jit("int64[:](int64[:], int64[:])", nopython=True)(_poly_divmod_jit)
        return cls._funcs["poly_divmod"]

    def _func_poly_evaluate(cls):
        if cls._funcs.get("poly_evaluate", None) is None:
            if cls.ufunc_mode == "python-calculate":
                cls._funcs["poly_evaluate"] = np.vectorize(cls._poly_evaluate_python, excluded=["coeffs"], otypes=[np.object_])
            else:
                global ADD_UFUNC, MULTIPLY_UFUNC
                ADD_UFUNC = cls._ufunc_add()
                MULTIPLY_UFUNC = cls._ufunc_multiply()
                assert cls.ufunc_target == "cpu"
                cls._funcs["poly_evaluate"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", nopython=True)(_poly_evaluate_jit)
        return cls._funcs["poly_evaluate"]

    def _func_poly_roots(cls):
        if cls._funcs.get("poly_roots", None) is None:
            if cls.ufunc_mode == "python-calculate":
                cls._funcs["poly_roots"] = cls._poly_roots_python
            else:
                global ORDER, PRIMITIVE_ELEMENT, ADD_UFUNC, MULTIPLY_UFUNC, POWER_UFUNC
                ORDER = cls.order
                PRIMITIVE_ELEMENT = int(cls.primitive_element)
                ADD_UFUNC = cls._ufunc_add()
                MULTIPLY_UFUNC = cls._ufunc_multiply()
                POWER_UFUNC = cls._ufunc_power()
                assert cls.ufunc_target == "cpu"
                cls._funcs["poly_roots"] = numba.jit("int64[:](int64[:], int64[:])", nopython=True)(_poly_roots_jit)
        return cls._funcs["poly_roots"]

    def _func_berlekamp_massey(cls):
        if cls._funcs.get("berlekamp_massey", None) is None:
            if cls.ufunc_mode == "python-calculate":
                cls._funcs["berlekamp_massey"] = cls._berlekamp_massey_python
            else:
                global ADD_UFUNC, SUBTRACT_UFUNC, MULTIPLY_UFUNC, RECIPROCAL_UFUNC
                ADD_UFUNC = cls._ufunc_add()
                SUBTRACT_UFUNC = cls._ufunc_subtract()
                MULTIPLY_UFUNC = cls._ufunc_multiply()
                RECIPROCAL_UFUNC = cls._ufunc_reciprocal()
                assert cls.ufunc_target == "cpu"
                cls._funcs["berlekamp_massey"] = numba.jit("int64[:](int64[:])", nopython=True)(_berlekamp_massey_jit)
        return cls._funcs["berlekamp_massey"]

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
            return _lapack_linalg(A, B, np.matmul, out=out)

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

        if cls._ufunc_mode != "python-calculate":
            A, B = A.astype(np.int64), B.astype(np.int64)

        C = cls._func_matmul()(A, B)

        if cls._ufunc_mode != "python-calculate":
            C = C.astype(dtype).view(field)

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
            raise ValueError(f"Operation 'convolve' currently only supports mode of 'full', not '{mode}'.")
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
                a, b = a.astype(np.int64), b.astype(np.int64)

            c = cls._func_convolve()(a, b)

            if cls._ufunc_mode != "python-calculate":
                c = c.astype(dtype).view(field)

            return c

    def _poly_divmod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        field = type(a)
        dtype = a.dtype
        q_degree = a.size - b.size
        r_degree = b.size - 1

        if cls._ufunc_mode != "python-calculate":
            a, b = a.astype(np.int64), b.astype(np.int64)

        qr = cls._func_poly_divmod()(a, b)

        if cls._ufunc_mode != "python-calculate":
            qr = qr.astype(dtype).view(field)

        return qr[0:q_degree + 1], qr[q_degree + 1:q_degree + 1 + r_degree + 1]

    def _poly_evaluate(cls, coeffs, x):
        assert isinstance(coeffs, cls) and isinstance(x, cls)
        assert coeffs.ndim == 1
        field = cls
        dtype = x.dtype
        x = np.atleast_1d(x)

        if cls.ufunc_mode == "python-calculate":
            # For object dtypes, call the vectorized classmethod
            y = cls._func_poly_evaluate()(coeffs=coeffs.view(np.ndarray), value=x.view(np.ndarray))
        else:
            # For integer dtypes, call the JIT-compiled gufunc
            y = cls._func_poly_evaluate()(coeffs, x, field.Zeros(x.shape), casting="unsafe")
            y = y.astype(dtype)
        y = y.view(field)

        if y.size == 1:
            y = y[0]

        return y

    def _poly_roots(cls, nonzero_degrees, nonzero_coeffs):
        assert isinstance(nonzero_coeffs, cls)
        field = cls
        dtype = nonzero_coeffs.dtype

        if cls._ufunc_mode != "python-calculate":
            nonzero_degrees, nonzero_coeffs = nonzero_degrees.astype(np.int64), nonzero_coeffs.astype(np.int64)

        roots = cls._func_poly_roots()(nonzero_degrees, nonzero_coeffs)

        if cls._ufunc_mode != "python-calculate":
            roots = roots.astype(dtype).view(field)

        idxs = np.argsort(roots)
        return roots[idxs]

    def _berlekamp_massey(cls, sequence):
        assert isinstance(sequence, cls)
        field = cls
        dtype = sequence.dtype

        if cls._ufunc_mode != "python-calculate":
            sequence = sequence.astype(np.int64)

        coeffs = cls._func_berlekamp_massey()(sequence)

        if cls._ufunc_mode != "python-calculate":
            coeffs = coeffs.astype(dtype).view(field)

        return coeffs

    ###############################################################################
    # Pure python implementation, operating on Galois field arrays (not integers),
    # for fields in ufunc_mode="python-calculate"
    ###############################################################################

    def _matmul_python(cls, A, B):
        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[-1] == B.shape[-2]
        M, N = A.shape[-2], B.shape[-1]
        C = cls.Zeros((M, N), dtype=A.dtype)

        for i in range(M):
            for j in range(N):
                C[i,j] = np.sum(A[i,:] * B[:,j])

        return C

    def _convolve_python(cls, a, b):
        c = cls.Zeros(a.size + b.size - 1, dtype=a.dtype)

        # Want a to be the shorter sequence
        if b.size < a.size:
            a, b = b, a

        for i in range(a.size):
            c[i:i + b.size] += a[i] * b

        return c

    def _poly_divmod_python(cls, a, b):
        # pylint: disable=unsupported-assignment-operation
        assert a.size >= b.size
        q_degree = a.size - b.size
        qr = cls(a)

        for i in range(0, q_degree + 1):
            if qr[i] > 0:
                q = qr[i] / b[0]
                qr[i:i + b.size] -= q*b
                qr[i] = q

        return qr

    def _poly_evaluate_python(cls, coeffs, value):
        result = coeffs[0]
        for j in range(1, coeffs.size):
            result = cls._add_python(coeffs[j], cls._multiply_python(result, value))
        return result

    def _poly_roots_python(cls, nonzero_degrees, nonzero_coeffs):
        lambda_vector = nonzero_coeffs
        alpha_vector = cls.primitive_element ** nonzero_degrees
        degree = max(nonzero_degrees)
        roots = []

        # Test if 0 is a root
        if 0 not in nonzero_degrees:
            roots.append(0)

        # Test if 1 is a root
        if np.sum(lambda_vector) == 0:
            roots.append(1)

        # Test if the powers of alpha are roots
        for i in range(1, cls.order - 1):
            lambda_vector *= alpha_vector
            if np.sum(lambda_vector) == 0:
                root = int(cls.primitive_element**i)
                roots.append(root)
            if len(roots) == degree:
                break

        return cls(roots)

    def _berlekamp_massey_python(cls, sequence):
        N = sequence.size

        s = sequence
        c = cls.Zeros(N)
        b = cls.Zeros(N)
        c[0] = 1  # The polynomial c(x) = 1
        b[0] = 1  # The polynomial b(x) = 1
        L = 0
        m = 1
        bb = cls(1)

        for n in range(0, N):
            d = np.sum(s[n - L:n + 1][::-1] * c[0:L + 1])

            if d == 0:
                m += 1
            elif 2*L <= n:
                t = np.copy(c)
                c[m:] -= d*np.reciprocal(bb)*b[0:N - m]
                L = n + 1 - L
                b = np.copy(t)
                bb = d
                m = 1
            else:
                c[m:] -= d*np.reciprocal(bb)*b[0:N - m]
                m += 1

        return c[0:L + 1][::-1]


###############################################################################
# JIT-compiled implementation of the specified functions
###############################################################################

def _matmul_jit(A, B):  # pragma: no cover
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[-1] == B.shape[-2]
    M, K = A.shape
    K, N = B.shape
    C = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i,j] = ADD_UFUNC(C[i,j], MULTIPLY_UFUNC(A[i,k], B[k,j]))
    return C


def _convolve_jit(a, b):  # pragma: no cover
    c = np.zeros(a.size + b.size - 1, dtype=a.dtype)

    for i in range(a.size):
        for j in range(b.size - 1, -1, -1):
            c[i + j] = ADD_UFUNC(c[i + j], MULTIPLY_UFUNC(a[i], b[j]))

    return c


def _poly_divmod_jit(a, b):  # pragma: no cover
    assert a.size >= b.size
    q_degree = a.size - b.size
    qr = np.copy(a)

    for i in range(0, q_degree + 1):
        if qr[i] > 0:
            q = DIVIDE_UFUNC(qr[i], b[0])
            for j in range(0, b.size):
                qr[i + j] = SUBTRACT_UFUNC(qr[i + j], MULTIPLY_UFUNC(q, b[j]))
            qr[i] = q

    return qr


def _poly_evaluate_jit(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_UFUNC(coeffs[j], MULTIPLY_UFUNC(results[i], values[i]))


def _poly_roots_jit(nonzero_degrees, nonzero_coeffs):
    N = nonzero_degrees.size
    lambda_vector = nonzero_coeffs
    alpha_vector = np.zeros(N, dtype=np.int64)
    for i in range(N):
        alpha_vector[i] = POWER_UFUNC(PRIMITIVE_ELEMENT, nonzero_degrees[i])
    degree = np.max(nonzero_degrees)
    roots = []

    # Test if 0 is a root
    if nonzero_degrees[-1] != 0:
        roots.append(0)

    # Test if 1 is a root
    _sum = 0
    for i in range(N):
        _sum = ADD_UFUNC(_sum, lambda_vector[i])
    if _sum == 0:
        roots.append(1)

    # Test if the powers of alpha are roots
    for i in range(1, ORDER - 1):
        _sum = 0
        for j in range(N):
            lambda_vector[j] = MULTIPLY_UFUNC(lambda_vector[j], alpha_vector[j])
            _sum = ADD_UFUNC(_sum, lambda_vector[j])
        if _sum == 0:
            root = POWER_UFUNC(PRIMITIVE_ELEMENT, i)
            roots.append(root)
        if len(roots) == degree:
            break

    return np.array(roots, dtype=np.int64)


def _berlekamp_massey_jit(sequence):
    N = sequence.size

    s = sequence
    c = np.zeros(N, dtype=np.int64)
    b = np.zeros(N, dtype=np.int64)
    c[0] = 1  # The polynomial c(x) = 1
    b[0] = 1  # The polynomial b(x) = 1
    L = 0
    m = 1
    bb = 1

    for n in range(0, N):
        d = 0
        for i in range(0, L + 1):
            d = ADD_UFUNC(d, MULTIPLY_UFUNC(s[n - i], c[i]))

        if d == 0:
            m += 1
        elif 2*L <= n:
            t = np.copy(c)
            d_bb = MULTIPLY_UFUNC(d, RECIPROCAL_UFUNC(bb))
            for i in range(m, N):
                c[i] = SUBTRACT_UFUNC(c[i], MULTIPLY_UFUNC(d_bb, b[i - m]))
            L = n + 1 - L
            b = np.copy(t)
            bb = d
            m = 1
        else:
            d_bb = MULTIPLY_UFUNC(d, RECIPROCAL_UFUNC(bb))
            for i in range(m, N):
                c[i] = SUBTRACT_UFUNC(c[i], MULTIPLY_UFUNC(d_bb, b[i - m]))
            m += 1

    return c[0:L + 1][::-1]
