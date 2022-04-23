"""
A module that contains Array mixin classes that override NumPy linear algebra functions. Additional functions not included
in NumPy are also included.
"""
import abc
from typing import Tuple, Optional

import numba
from numba import int64
import numpy as np

from ._array import Array, DTYPES
from ._function import RingFunctions, FieldFunctions


class RingLinalgFunctions(RingFunctions, abc.ABC):
    """
    A mixin base class that overrides NumPy linear algebra functions to perform ring arithmetic (+, -, *), using *only* explicit
    calculation.
    """

    _OVERRIDDEN_FUNCTIONS = {**RingFunctions._OVERRIDDEN_FUNCTIONS, **{
        np.dot: "_dot",
        np.vdot: "_vdot",
        np.inner: "_inner",
        np.outer: "_outer",
        # np.tensordot: "_tensordot",
    }}

    ###############################################################################
    # ndarray method overrides
    ###############################################################################

    def dot(self, b, out=None):
        # The `np.dot(a, b)` ufunc is also available as `a.dot(b)`. Need to override this method for consistent results.
        return type(self)._dot(self, b, out=out)

    ###############################################################################
    # Helper methods
    ###############################################################################

    @classmethod
    def _lapack_linalg(cls, a: Array, b: Array, function, out=None, n_sum=None) -> Array:
        """
        In prime fields GF(p), it's much more efficient to use LAPACK/BLAS implementations of linear algebra
        and then reduce modulo p rather than compute manually.
        """
        assert cls._is_prime_field

        # Determine the return data-type which is the minimum of the two inputs' data-types
        if np.object_ in [a.dtype, b.dtype]:
            return_dtype = np.object_
        else:
            return_dtype = a.dtype if np.iinfo(a.dtype).max < np.iinfo(b.dtype).max else b.dtype

        a = a.view(np.ndarray)
        b = b.view(np.ndarray)

        # Determine the minimum dtype to hold the entire product and summation without overflowing
        if n_sum is None:
            n_sum = 1 if len(a.shape) == 0 else max(a.shape)
        max_value = n_sum * (cls.characteristic - 1)**2
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= max_value]
        dtype = np.object_ if len(dtypes) == 0 else dtypes[0]
        a = a.astype(dtype)
        b = b.astype(dtype)

        # Compute result using native NumPy LAPACK/BLAS implementation
        if function in [np.inner, np.vdot]:
            # These functions don't have and `out` keyword argument
            c = function(a, b)
        else:
            c = function(a, b, out=out)
        c = c % cls.characteristic  # Reduce the result mod p

        if np.isscalar(c):
            # TODO: Sometimes the scalar c is a float?
            c = cls(int(c), dtype=return_dtype)
        else:
            c = cls._view(c.astype(return_dtype))

        return c

    ###############################################################################
    # Matrix/vector products
    ###############################################################################

    @classmethod
    def _dot(cls, a: Array, b: Array, out=None) -> Array:
        """
        https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        """
        if not type(a) is type(b):
            raise TypeError(f"Operation 'dot' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

        if cls._is_prime_field:
            return cls._lapack_linalg(a, b, np.dot, out=out)

        if a.ndim == 0 or b.ndim == 0:
            return a * b
        elif a.ndim == 1 and b.ndim == 1:
            return np.sum(a * b)
        elif a.ndim == 2 and b.ndim == 2:
            return np.matmul(a, b, out=out)
        elif a.ndim >= 2 and b.ndim == 1:
            return np.sum(a * b, axis=-1, out=out)
        # elif a.dnim >= 2 and b.ndim >= 2:
        else:
            raise NotImplementedError("Currently 'dot' is only supported up to 2-D matrices. Please open a GitHub issue at https://github.com/mhostetter/galois/issues.")

    @classmethod
    def _vdot(cls, a: Array, b: Array) -> Array:
        """
        https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
        """
        if not type(a) is type(b):
            raise TypeError(f"Operation 'vdot' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

        if cls._is_prime_field:
            return cls._lapack_linalg(a, b, np.vdot)

        a = a.flatten()
        b = b.flatten().reshape(a.shape)  # This is done to mimic NumPy's error scenarios

        return np.sum(a * b)

    @classmethod
    def _inner(cls, a: Array, b: Array) -> Array:
        """
        https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner
        """
        if not type(a) is type(b):
            raise TypeError(f"Operation 'inner' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

        if cls._is_prime_field:
            return cls._lapack_linalg(a, b, np.inner)

        if a.ndim == 0 or b.ndim == 0:
            return a * b
        if not a.shape[-1] == b.shape[-1]:
            raise ValueError(f"Operation 'inner' requires `a` and `b` to have the same last dimension, not {a.shape} and {b.shape}.")

        return np.sum(a * b, axis=-1)

    @classmethod
    def _outer(cls, a: Array, b: Array, out=None) -> Array:
        """
        https://numpy.org/doc/stable/reference/generated/numpy.outer.html#numpy.outer
        """
        if not type(a) is type(b):
            raise TypeError(f"Operation 'outer' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

        if cls._is_prime_field:
            return cls._lapack_linalg(a, b, np.outer, out=out, n_sum=1)
        else:
            return np.multiply.outer(a.ravel(), b.ravel(), out=out)

    ###############################################################################
    # Matrix inversions, solutions, rank, etc
    ###############################################################################

    @classmethod
    def _triangular_det(cls, A: Array) -> Array:
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
        idxs = np.arange(0, A.shape[0])
        return np.multiply.reduce(A[idxs,idxs])

    @classmethod
    def _det(cls, A: Array) -> Array:
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")

        n = A.shape[0]

        if n == 2:
            return A[0,0]*A[1,1] - A[0,1]*A[1,0]
        elif n == 3:
            return A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0])
        else:
            # TODO: Fix this
            P, L, U, N_permutations = cls._plu_decompose(A)  # pylint: disable=no-member
            P = P.T  # Convert row permutation matrix into column permutation matrix
            det_P = (-cls(1)) ** N_permutations
            det_L = cls._triangular_det(L)
            det_U = cls._triangular_det(U)
            return det_P * det_L * det_U

    ###############################################################################
    # Matrix multiplication
    ###############################################################################

    @classmethod
    def _matmul(cls, A: Array, B: Array, out=None, **kwargs) -> Array:  # pylint: disable=unused-argument
        if not type(A) is type(B):
            raise TypeError(f"Operation 'matmul' requires both arrays be in the same Galois field, not {type(A)} and {type(B)}.")
        if not (A.ndim >= 1 and B.ndim >= 1):
            raise ValueError(f"Operation 'matmul' requires both arrays have dimension at least 1, not {A.ndim}-D and {B.ndim}-D.")
        if not (A.ndim <= 2 and B.ndim <= 2):
            raise ValueError("Operation 'matmul' currently only supports matrix multiplication up to 2-D. If you would like matrix multiplication of N-D arrays, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")
        dtype = A.dtype

        if cls._is_prime_field:
            return cls._lapack_linalg(A, B, np.matmul, out=out)

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
            C = cls._function("matmul")(A, B, add, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            C = C.astype(dtype)
        else:
            A = A.view(np.ndarray)
            B = B.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            C = cls._function("matmul")(A, B, add, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
        C = cls._view(C)

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

    _MATMUL_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](
        int64[:,:],
        int64[:,:],
        RingFunctions._BINARY_CALCULATE_SIG,
        RingFunctions._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

    @staticmethod
    @numba.extending.register_jitable
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


class FieldLinalgFunctions(RingLinalgFunctions, FieldFunctions, abc.ABC):
    """
    A mixin base class that overrides NumPy linear algebra functions to perform field arithmetic (+, -, *, /), using *only* explicit
    calculation.
    """

    _OVERRIDDEN_FUNCTIONS = {**RingLinalgFunctions._OVERRIDDEN_FUNCTIONS, **{
        np.linalg.det: "_det",
        np.linalg.matrix_rank: "_matrix_rank",
        np.linalg.solve: "_solve",
        np.linalg.inv: "_inv",
    }}

    ###############################################################################
    # Matrix decomposition routines
    ###############################################################################

    @classmethod
    def _row_reduce(cls, A: Array, ncols: Optional[int] = None) -> Tuple[Array, int]:
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices can be converted to reduced row echelon form, not {A.ndim}-D.")

        ncols = A.shape[1] if ncols is None else ncols
        A_rre = A.copy()
        p = 0  # The pivot

        for j in range(ncols):
            # Find a pivot in column `j` at or below row `p`
            idxs = np.nonzero(A_rre[p:,j])[0]
            if idxs.size == 0:
                continue
            i = p + idxs[0]  # Row with a pivot

            # Swap row `p` and `i`. The pivot is now located at row `p`.
            A_rre[[p,i],:] = A_rre[[i,p],:]

            # Force pivot value to be 1
            A_rre[p,:] /= A_rre[p,j]

            # Force zeros above and below the pivot
            idxs = np.nonzero(A_rre[:,j])[0].tolist()
            idxs.remove(p)
            A_rre[idxs,:] -= np.multiply.outer(A_rre[idxs,j], A_rre[p,:])

            p += 1
            if p == A_rre.shape[0]:
                break

        return A_rre, p

    @classmethod
    def _lu_decompose(cls, A: Array) -> Tuple[Array, Array]:
        if not A.ndim == 2:
            raise ValueError(f"Argument `A` must be a 2-D matrix, not have shape {A.shape}.")

        n = A.shape[0]
        Ai = A.copy()
        L = cls.Identity(n)

        for i in range(0, n-1):
            if Ai[i,i] == 0:
                idxs = np.nonzero(Ai[i:,i])[0]  # The first non-zero entry in column `i` below row `i`
                if idxs.size == 0:  # pylint: disable=no-else-continue
                    L[i,i] = 1
                    continue
                else:
                    raise ValueError("The LU decomposition of `A` does not exist. Use the LUP decomposition instead.")

            l = Ai[i+1:,i] / Ai[i,i]
            Ai[i+1:,:] -= np.multiply.outer(l, Ai[i,:])
            L[i+1:,i] = l

        U = Ai

        return L, U

    @classmethod
    def _plu_decompose(cls, A: Array) -> Tuple[Array, Array, Array, int]:
        if not A.ndim == 2:
            raise ValueError(f"Argument `A` must be a 2-D matrix, not have shape {A.shape}.")

        n = A.shape[0]
        Ai = A.copy()
        L = cls.Zeros((n,n))
        P = cls.Identity(n)  # Row permutation matrix
        N_permutations = 0  # Number of permutations

        for i in range(0, n-1):
            if Ai[i,i] == 0:
                idxs = np.nonzero(Ai[i:,i])[0]  # The first non-zero entry in column `i` below row `i`
                if idxs.size == 0:
                    L[i,i] = 1
                    continue
                j = i + idxs[0]

                # Swap rows `i` and `j`
                P[[i,j],:] = P[[j,i],:]
                Ai[[i,j],:] = Ai[[j,i],:]
                L[[i,j],:] = L[[j,i],:]
                N_permutations += 1

            l = Ai[i+1:,i] / Ai[i,i]
            Ai[i+1:,:] -= np.multiply.outer(l, Ai[i,:])  # Zero out rows below row `i`
            L[i,i] = 1  # Set 1 on the diagonal
            L[i+1:,i] = l

        L[-1,-1] = 1  # Set the final diagonal to 1
        U = Ai

        # NOTE: Return column permutation matrix
        return P.T, L, U, N_permutations

    ###############################################################################
    # Matrix inversions, solutions, rank, etc
    ###############################################################################

    @classmethod
    def _matrix_rank(cls, A: Array) -> int:
        A_rre, _ = cls._row_reduce(A)
        rank = np.sum(~np.all(A_rre == 0, axis=1))

        return rank

    @classmethod
    def _inv(cls, A: Array) -> Array:
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")

        n = A.shape[0]
        I = cls.Identity(n, dtype=A.dtype)

        # Concatenate A and I to get the matrix AI = [A | I]
        AI = np.concatenate((A, I), axis=-1)

        # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
        AI_rre, _ = cls._row_reduce(AI, ncols=n)

        # The rank is the number of non-zero rows of the row reduced echelon form
        rank = np.sum(~np.all(AI_rre[:,0:n] == 0, axis=1))
        if not rank == n:
            raise np.linalg.LinAlgError(f"Argument `A` is singular and not invertible because it does not have full rank of {n}, but rank of {rank}.")

        A_inv = AI_rre[:,-n:]

        return A_inv

    @classmethod
    def _solve(cls, A: Array, b: Array) -> Array:
        if not type(A) is type(b):
            raise TypeError(f"Arguments `A` and `b` must be of the same FieldArray subclass, not {type(A)} and {type(b)}.")
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
        if not b.ndim in [1, 2]:
            raise np.linalg.LinAlgError(f"Argument `b` must have dimension equal to A or one less, not {b.ndim}.")
        if not A.shape[-1] == b.shape[0]:
            raise np.linalg.LinAlgError(f"The last dimension of `A` must equal the first dimension of `b`, not {A.shape} and {b.shape}.")

        A_inv = cls._inv(A)
        x = A_inv @ b

        return x

    ###############################################################################
    # Vector spaces
    ###############################################################################

    @classmethod
    def _row_space(cls, A: Array) -> Array:
        """
        R(A) = C(A^T)
        """
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a row space, not {A.ndim}-D.")

        A_rre, _ = cls._row_reduce(A)
        rank = np.sum(~np.all(A_rre == 0, axis=1))
        R = A_rre[0:rank,:]

        return R

    @classmethod
    def _column_space(cls, A: Array) -> Array:
        """
        C(A) = R(A^T)
        """
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a column space, not {A.ndim}-D.")

        return cls._row_space(A.T)

    @classmethod
    def _left_null_space(cls, A: Array) -> Array:
        """
        x = LN(A) = N(A^T)
        x A = 0
        """
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a left null space, not {A.ndim}-D.")

        m, n = A.shape
        I = cls.Identity(m, dtype=A.dtype)

        # Concatenate A and I to get the matrix AI = [A | I]
        AI = np.concatenate((A, I), axis=-1)

        # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
        AI_rre, p = cls._row_reduce(AI, ncols=n)

        # Row reduce the left null space so that it begins with an I
        LN = AI_rre[p:,n:]
        LN, _ = cls._row_reduce(LN)

        return LN

    @classmethod
    def _null_space(cls, A: Array) -> Array:
        """
        x = N(A) = LN(A^T)
        A x = 0
        """
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a null space, not {A.ndim}-D.")

        return cls._left_null_space(A.T)
