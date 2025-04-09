"""
A module that contains Array mixin classes that override NumPy linear algebra functions. Additional functions not
included in NumPy are also included.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numba
import numpy as np
from numba import int64

from .._helper import verify_isinstance
from ._function import Function, FunctionMixin

if TYPE_CHECKING:
    from ._array import Array


def _lapack_linalg(field: Type[Array], a: Array, b: Array, function, out=None, n_sum=None) -> Array:
    """
    In prime fields GF(p), it's much more efficient to use LAPACK/BLAS implementations of linear algebra
    and then reduce modulo p rather than compute manually.
    """
    assert field._is_prime_field

    # Determine the return data-type which is the minimum of the two inputs' data-types
    if np.object_ in [a.dtype, b.dtype]:
        return_dtype = np.object_
    else:
        return_dtype = a.dtype if np.iinfo(a.dtype).max < np.iinfo(b.dtype).max else b.dtype

    a = a.view(np.ndarray)
    b = b.view(np.ndarray)

    # Determine the maximum possible integer summation. Find the smallest floating-point data type that can hold that
    # integer value. This enables the linear algebra to be computed using BLAS, which is faster than integer arithmetic.
    # The maximum integer that can be represented in a float64 is 2**(np.finfo(np.float64).nmant + 1). If
    # floating-point arithmetic is not possible, then np.int64 or np.object_ is used.
    if n_sum is None:
        n_sum = 1 if len(a.shape) == 0 else max(a.shape)
    max_value = n_sum * (field.characteristic - 1) ** 2
    if max_value <= 2 ** (np.finfo(np.float32).nmant + 1):
        dtype = np.float32
    elif max_value <= 2 ** (np.finfo(np.float64).nmant + 1):
        dtype = np.float64
    elif max_value <= np.iinfo(np.int64).max:
        dtype = np.int64
    else:
        dtype = np.object_
    a = a.astype(dtype)
    b = b.astype(dtype)

    # Compute result using native NumPy LAPACK/BLAS implementation
    if function in [np.inner, np.vdot]:
        # These functions don't have and `out` keyword argument
        c = function(a, b)
    else:
        c = function(a, b, out=out)
    c = c % field.characteristic  # Reduce the result mod p

    if np.isscalar(c):
        # TODO: Sometimes the scalar c is a float?
        c = field(int(c), dtype=return_dtype)
    else:
        c = field._view(c.astype(return_dtype))

    return c


###############################################################################
# Matrix products
###############################################################################


class dot_jit(Function):
    """
    Computes the dot product of two arrays.

    References:
        - https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    """

    def __call__(self, a: Array, b: Array, out=None) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)

        if self.field._is_prime_field:
            return _lapack_linalg(self.field, a, b, np.dot, out=out)

        if a.ndim == 0 or b.ndim == 0:
            dot = a * b
        elif a.ndim == 1 and b.ndim == 1:
            dot = np.sum(a * b)
        elif a.ndim == 2 and b.ndim == 2:
            dot = np.matmul(a, b, out=out)
        elif a.ndim >= 2 and b.ndim == 1:
            dot = np.sum(a * b, axis=-1, out=out)
        # elif a.dnim >= 2 and b.ndim >= 2:
        else:
            raise NotImplementedError(
                "Currently 'dot' is only supported up to 2-D matrices. "
                "Please open a GitHub issue at https://github.com/mhostetter/galois/issues."
            )

        return dot


class vdot_jit(Function):
    """
    Computes the vector dot product of two arrays.

    References:
        - https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)

        if self.field._is_prime_field:
            return _lapack_linalg(self.field, a, b, np.vdot)

        a = a.flatten()
        b = b.flatten().reshape(a.shape)  # This is done to mimic NumPy's error scenarios

        return np.sum(a * b)


class inner_jit(Function):
    """
    Computes the inner product of two arrays.

    References:
        - https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)

        if self.field._is_prime_field:
            return _lapack_linalg(self.field, a, b, np.inner)

        if a.ndim == 0 or b.ndim == 0:
            return a * b
        if not a.shape[-1] == b.shape[-1]:
            raise ValueError(
                f"Operation 'inner' requires `a` and `b` to have the same last dimension, not {a.shape} and {b.shape}."
            )

        return np.sum(a * b, axis=-1)


class outer_jit(Function):
    """
    Computes the outer product of two arrays.

    References:
        - https://numpy.org/doc/stable/reference/generated/numpy.outer.html#numpy.outer
    """

    def __call__(self, a: Array, b: Array, out=None) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)

        if self.field._is_prime_field:
            return _lapack_linalg(self.field, a, b, np.outer, out=out, n_sum=1)

        return np.multiply.outer(a.ravel(), b.ravel(), out=out)


class matmul_jit(Function):
    """
    Computes the matrix multiplication of two matrices.

    np.matmul is technically a NumPy ufunc, so this JIT function will be called from matmul_ufunc.

    References:
        - https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul
    """

    def __call__(self, A: Array, B: Array, out=None, **kwargs) -> Array:
        verify_isinstance(A, self.field)
        verify_isinstance(B, self.field)
        if not (A.ndim >= 1 and B.ndim >= 1):
            raise ValueError(
                f"Operation 'matmul' requires both arrays have dimension at least 1, not {A.ndim}-D and {B.ndim}-D."
            )
        if not (A.ndim <= 2 and B.ndim <= 2):
            raise ValueError(
                "Operation 'matmul' currently only supports matrix multiplication up to 2-D. "
                "Please open a GitHub issue at https://github.com/mhostetter/galois/issues."
            )
        dtype = A.dtype

        if self.field._is_prime_field:
            return _lapack_linalg(self.field, A, B, np.matmul, out=out)

        prepend, append = False, False
        if A.ndim == 1:
            A = A.reshape((1, A.size))
            prepend = True
        if B.ndim == 1:
            B = B.reshape((B.size, 1))
            append = True

        if not A.shape[-1] == B.shape[-2]:
            raise ValueError(
                f"Operation 'matmul' requires the last dimension of 'A' to match the second-to-last dimension of 'B', "
                f"not {A.shape} and {B.shape}."
            )

        # if A.ndim > 2 and B.ndim == 2:
        #     new_shape = list(A.shape[:-2]) + list(B.shape)
        #     B = np.broadcast_to(B, new_shape)
        # if B.ndim > 2 and A.ndim == 2:
        #     new_shape = list(B.shape[:-2]) + list(A.shape)
        #     A = np.broadcast_to(A, new_shape)

        if self.field.ufunc_mode != "python-calculate":
            C = self.jit(A.astype(np.int64), B.astype(np.int64))
            C = C.astype(dtype)
        else:
            C = self.python(A.view(np.ndarray), B.view(np.ndarray))
        C = self.field._view(C)

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

    def set_globals(self):
        global ADD, MULTIPLY
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:, :](int64[:, :], int64[:, :]))
    _PARALLEL = True

    @staticmethod
    def implementation(A, B):
        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[-1] == B.shape[-2]

        M, K = A.shape
        K, N = B.shape
        C = np.zeros((M, N), dtype=A.dtype)
        for i in numba.prange(M):
            for j in numba.prange(N):
                for k in range(K):
                    C[i, j] = ADD(C[i, j], MULTIPLY(A[i, k], B[k, j]))

        return C


###############################################################################
# Matrix decomposition routines
###############################################################################


class row_reduce_jit(Function):
    """
    Converts the matrix into its row-reduced echelon form using Gaussian elimination.
    """

    def __call__(self, A: Array, ncols: int | None = None) -> tuple[Array, int]:
        verify_isinstance(A, self.field)
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices can be converted to reduced row echelon form, not {A.ndim}-D.")

        ncols = A.shape[1] if ncols is None else ncols
        A_rre = A.copy()
        p = 0  # The pivot

        for j in range(ncols):
            # Find a pivot in column `j` at or below row `p`
            idxs = np.nonzero(A_rre[p:, j])[0]
            if idxs.size == 0:
                continue
            i = p + idxs[0]  # Row with a pivot

            # Swap row `p` and `i`. The pivot is now located at row `p`.
            A_rre[[p, i], :] = A_rre[[i, p], :]

            # Force pivot value to be 1
            A_rre[p, :] /= A_rre[p, j]

            # Force zeros above and below the pivot
            idxs = np.nonzero(A_rre[:, j])[0].tolist()
            idxs.remove(p)
            A_rre[idxs, :] -= np.multiply.outer(A_rre[idxs, j], A_rre[p, :])

            p += 1
            if p == A_rre.shape[0]:
                break

        return A_rre, p


class lu_decompose_jit(Function):
    """
    Decomposes the matrix into its LU decomposition.
    """

    def __call__(self, A: Array) -> tuple[Array, Array]:
        verify_isinstance(A, self.field)
        if not A.ndim == 2:
            raise ValueError(f"Argument 'A' must be a 2-D matrix, not have shape {A.shape}.")

        m = A.shape[0]
        Ai = A.copy()
        L = self.field.Identity(m)

        for i in range(0, m - 1):
            if Ai[i, i] == 0:
                idxs = np.nonzero(Ai[i:, i])[0]  # The first non-zero entry in column `i` below row `i`
                if idxs.size == 0:
                    L[i, i] = 1
                    continue
                else:
                    raise ValueError("The LU decomposition of 'A' does not exist. Use the PLU decomposition instead.")

            l = Ai[i + 1 :, i] / Ai[i, i]
            Ai[i + 1 :, :] -= np.multiply.outer(l, Ai[i, :])
            L[i + 1 :, i] = l

        U = Ai

        return L, U


class plu_decompose_jit(Function):
    """
    Decomposes the matrix into its PLU decomposition.
    """

    def __call__(self, A: Array) -> tuple[Array, Array, Array, int]:
        verify_isinstance(A, self.field)
        if not A.ndim == 2:
            raise ValueError(f"Argument 'A' must be a 2-D matrix, not have shape {A.shape}.")

        m, n = A.shape
        Ai = A.copy()
        L = self.field.Zeros((m, m))
        P = self.field.Identity(m)  # Row permutation matrix
        N_permutations = 0  # Number of permutations

        for i in range(0, min(m, n)):
            if Ai[i, i] == 0:
                idxs = np.nonzero(Ai[i:, i])[0]  # The first non-zero entry in column `i` below row `i`
                if idxs.size == 0:
                    L[i, i] = 1
                    continue
                j = i + idxs[0]

                # Swap rows `i` and `j`
                P[[i, j], :] = P[[j, i], :]
                Ai[[i, j], :] = Ai[[j, i], :]
                L[[i, j], :] = L[[j, i], :]
                N_permutations += 1

            l = Ai[i + 1 :, i] / Ai[i, i]
            Ai[i + 1 :, :] -= np.multiply.outer(l, Ai[i, :])  # Zero out rows below row `i`
            L[i, i] = 1  # Set 1 on the diagonal
            L[i + 1 :, i] = l

        L[-1, -1] = 1  # Set the final diagonal to 1
        U = Ai

        # NOTE: Return column permutation matrix
        return P.T, L, U, N_permutations


###############################################################################
# Matrix inversions, solutions, rank, etc
###############################################################################


class triangular_det_jit(Function):
    """
    Computes the determinant of a triangular square matrix.
    """

    def __call__(self, A: Array) -> Array:
        verify_isinstance(A, self.field)
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument 'A' must be square, not {A.shape}.")
        idxs = np.arange(0, A.shape[0])
        return np.multiply.reduce(A[idxs, idxs])


class det_jit(Function):
    """
    Computes the determinant of a square matrix.
    """

    def __call__(self, A: Array) -> Array:
        verify_isinstance(A, self.field)
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument 'A' must be square, not {A.shape}.")

        n = A.shape[0]

        if n == 2:
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        elif n == 3:
            det = (
                A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
                - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
                + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
            )
        else:
            P, L, U, N_permutations = plu_decompose_jit(self.field)(A)
            P = P.T  # Convert row permutation matrix into column permutation matrix
            det_P = (-self.field(1)) ** N_permutations
            det_L = triangular_det_jit(self.field)(L)
            det_U = triangular_det_jit(self.field)(U)
            det = det_P * det_L * det_U

        return det


###############################################################################
# Matrix inversions, solutions, rank, etc
###############################################################################


class matrix_rank_jit(Function):
    """
    Computes the rank of the square matrix.
    """

    def __call__(self, A: Array) -> int:
        verify_isinstance(A, self.field)
        A_rre, _ = row_reduce_jit(self.field)(A)
        rank = np.sum(~np.all(A_rre == 0, axis=1))
        rank = int(rank)
        return rank


class inv_jit(Function):
    """
    Computes the inverse of the square matrix.
    """

    def __call__(self, A: Array) -> Array:
        verify_isinstance(A, self.field)
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument 'A' must be square, not {A.shape}.")

        n = A.shape[0]
        I = self.field.Identity(n, dtype=A.dtype)

        # Concatenate A and I to get the matrix AI = [A | I]
        AI = np.concatenate((A, I), axis=-1)

        # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
        AI_rre, _ = row_reduce_jit(self.field)(AI, ncols=n)

        # The rank is the number of non-zero rows of the row reduced echelon form
        rank = np.sum(~np.all(AI_rre[:, 0:n] == 0, axis=1))
        if not rank == n:
            raise np.linalg.LinAlgError(
                f"Argument 'A' is singular and not invertible because it does not have full rank of {n}, "
                f"but rank of {rank}."
            )

        A_inv = AI_rre[:, -n:]

        return A_inv


class solve_jit(Function):
    """
    Solves the linear system Ax = b.
    """

    def __call__(self, A: Array, b: Array) -> Array:
        verify_isinstance(A, self.field)
        verify_isinstance(b, self.field)
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument 'A' must be square, not {A.shape}.")
        if not b.ndim in [1, 2]:
            raise np.linalg.LinAlgError(f"Argument 'b' must have dimension equal to 'A' or one less, not {b.ndim}.")
        if not A.shape[-1] == b.shape[0]:
            raise np.linalg.LinAlgError(
                f"The last dimension of 'A' must equal the first dimension of 'b', not {A.shape} and {b.shape}."
            )

        A_inv = inv_jit(self.field)(A)
        x = A_inv @ b

        return x


###############################################################################
# Array mixin class
###############################################################################


class LinalgFunctionMixin(FunctionMixin):
    """
    A mixin base class that overrides NumPy linear algebra functions to perform self.field arithmetic (+, -, *, /),
    using *only* explicit calculation.
    """

    _OVERRIDDEN_FUNCTIONS = {
        **FunctionMixin._OVERRIDDEN_FUNCTIONS,
        **{
            np.dot: "_dot",
            np.vdot: "_vdot",
            np.inner: "_inner",
            np.outer: "_outer",
            # np.tensordot: "_tensordot",
            np.linalg.det: "_det",
            np.linalg.matrix_rank: "_matrix_rank",
            np.linalg.solve: "_solve",
            np.linalg.inv: "_inv",
        },
    }

    _dot: Function
    _vdot: Function
    _inner: Function
    _outer: Function
    _det: Function
    _matrix_rank: Function
    _solve: Function
    _inv: Function

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._dot = dot_jit(cls)
        cls._vdot = vdot_jit(cls)
        cls._inner = inner_jit(cls)
        cls._outer = outer_jit(cls)
        cls._det = det_jit(cls)
        cls._matrix_rank = matrix_rank_jit(cls)
        cls._solve = solve_jit(cls)
        cls._inv = inv_jit(cls)
