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

    if dtype in [np.float32, np.float64]:
        # Performing the modulo operation on an int is faster than a float. Convert back to int64 for the time
        # savings, if possible.
        c = c.astype(np.int64)

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
                f"Operation 'inner' requires 'a' and 'b' to have the same last dimension, not {a.shape} and {b.shape}."
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
        dtype = A.dtype

        if self.field._is_prime_field:
            return _lapack_linalg(self.field, A, B, np.matmul, out=out)

        # Handle 1-D semantics like np.matmul
        A_was_vec = A.ndim == 1
        B_was_vec = B.ndim == 1
        if A_was_vec:
            # (k,) -> (1, k), later we may drop the "M" dimension
            A = A.reshape(1, A.size)
        if B_was_vec:
            # (k,) -> (k, 1), later we may drop the "N" dimension
            B = B.reshape(B.size, 1)

        if A.ndim < 2 or B.ndim < 2:
            # After promoting 1-D to 2-D, both must be at least 2-D.
            raise ValueError(
                "Operation 'matmul' requires both arrays be at least 1-D. This should not occur after 1-D promotion."
            )
        M, K = A.shape[-2], A.shape[-1]
        K2, N = B.shape[-2], B.shape[-1]
        if K != K2:
            raise ValueError(
                f"Operation 'matmul' requires the last dimension of 'A' to match the "
                f"second-to-last dimension of 'B', not {A.shape} and {B.shape}."
            )

        # Broadcast batch dimensions
        batch_A = A.shape[:-2]
        batch_B = B.shape[:-2]
        try:
            # NumPy >= 1.20
            batch_shape = np.broadcast_shapes(batch_A, batch_B)
        except AttributeError:
            # Fallback for older NumPy
            batch_shape = np.broadcast(np.empty(batch_A), np.empty(batch_B)).shape
        A_b = np.broadcast_to(A, batch_shape + (M, K))
        B_b = np.broadcast_to(B, batch_shape + (K, N))

        # Flatten batch dimensions so the kernel sees (BATCH, M, K) and (BATCH, K, N)
        if batch_shape:
            BATCH = int(np.prod(batch_shape))
        else:
            BATCH = 1
            batch_shape = ()  # Make sure this is a tuple for later
        A_flat = A_b.reshape(BATCH, M, K)
        B_flat = B_b.reshape(BATCH, K, N)

        if self.field.ufunc_mode != "python-calculate":
            C_flat = self.jit(A_flat.astype(np.int64), B_flat.astype(np.int64))
            C_flat = C_flat.astype(dtype)
        else:
            C_flat = self.python(A_flat.view(np.ndarray), B_flat.view(np.ndarray))
        C = C_flat.reshape(batch_shape + (M, N))
        C = self.field._view(C)

        # Apply np.matmul-style 1-D squeezing rules
        if A_was_vec and B_was_vec:
            # (k,) @ (k,) -> scalar with batch dimensions (here usually no batch)
            final_shape = batch_shape
        elif A_was_vec and not B_was_vec:
            # (k,) @ (..., k, N) -> (..., N)
            final_shape = batch_shape + (N,)
        elif not A_was_vec and B_was_vec:
            # (..., M, K) @ (K,) -> (..., M)
            final_shape = batch_shape + (M,)
        else:
            # (..., M, K) @ (..., K, N) -> (..., M, N)
            final_shape = batch_shape + (M, N)
        C = C.reshape(final_shape)

        if out is not None:
            # NumPy ufuncs pass out as a 1-tuple; galois seems to mirror that.
            assert isinstance(out, tuple) and len(out) == 1
            out_arr = out[0]
            out_arr[...] = C[...]
            return out_arr

        return C

    def set_globals(self):
        global ADD, MULTIPLY
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:, :, :](int64[:, :, :], int64[:, :, :]))
    _PARALLEL = True

    @staticmethod
    def implementation(A, B):
        # A: (BATCH, M, K)
        # B: (BATCH, K, N)
        assert A.ndim == 3 and B.ndim == 3

        BATCH, M, K = A.shape
        BATCH2, K2, N = B.shape
        assert BATCH == BATCH2
        assert K == K2

        C = np.zeros((BATCH, M, N), dtype=A.dtype)

        # Parallelize across batch dimension (and optionally one of M/N if you want)
        for b in numba.prange(BATCH):
            for i in range(M):
                for j in range(N):
                    acc = 0
                    for k in range(K):
                        acc = ADD(acc, MULTIPLY(A[b, i, k], B[b, k, j]))
                    C[b, i, j] = acc

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
