"""
A module that contains linear algebra routines over Galois fields.
"""
import numpy as np

from ._dtypes import DTYPES


def _lapack_linalg(a, b, function, out=None, n_sum=None):
    """
    In prime fields GF(p), it's much more efficient to use LAPACK/BLAS implementations of linear algebra
    and then reduce modulo p rather than compute manually.
    """
    assert type(a).is_prime_field
    field = type(a)
    characteristic = field.characteristic

    # Determine the return data-type which is the minimum of the two inputs' data-types
    if a.dtype == np.object_ or b.dtype == np.object_:
        return_dtype = np.object_
    else:
        return_dtype = a.dtype if np.iinfo(a.dtype).max < np.iinfo(b.dtype).max else b.dtype

    a = a.view(np.ndarray)
    b = b.view(np.ndarray)

    # Determine the minimum dtype to hold the entire product and summation without overflowing
    if n_sum is None:
        n_sum = 1 if len(a.shape) == 0 else max(a.shape)
    max_value = n_sum * (characteristic - 1)**2
    dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= max_value]
    dtype = np.object_ if len(dtypes) == 0 else dtypes[0]
    a = a.astype(dtype)
    b = b.astype(dtype)

    # Compute result using native numpy LAPACK/BLAS implementation
    if function in [np.inner, np.vdot]:
        # These functions don't have and `out` keyword argument
        c = function(a, b)
    else:
        c = function(a, b, out=out)
    c = c % characteristic  # Reduce the result mod p

    if np.isscalar(c):
        # TODO: Sometimes the scalar c is a float?
        c = field(int(c), dtype=return_dtype)
    else:
        c = c.astype(return_dtype).view(field)

    return c


###############################################################################
# Matrix/vector products
###############################################################################

def dot(a, b, out=None):  # pylint: disable=unused-argument
    """
    https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    """
    if not type(a) is type(b):
        raise TypeError(f"Operation 'dot' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

    if type(a).is_prime_field:
        return _lapack_linalg(a, b, np.dot, out=out)

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


def vdot(a, b):
    """
    https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
    """
    if not type(a) is type(b):
        raise TypeError(f"Operation 'vdot' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

    if type(a).is_prime_field:
        return _lapack_linalg(a, b, np.vdot)

    a = a.flatten()
    b = b.flatten().reshape(a.shape)  # This is done to mimic numpy's error scenarios

    return np.sum(a * b)


def inner(a, b):
    """
    https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner
    """
    if not type(a) is type(b):
        raise TypeError(f"Operation 'inner' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

    if type(a).is_prime_field:
        return _lapack_linalg(a, b, np.inner)

    if a.ndim == 0 or b.ndim == 0:
        return a * b
    if not a.shape[-1] == b.shape[-1]:
        raise ValueError(f"Operation 'inner' requires `a` and `b` to have the same last dimension, not {a.shape} and {b.shape}.")

    return np.sum(a * b, axis=-1)


def outer(a, b, out=None):  # pylint: disable=unused-argument
    """
    https://numpy.org/doc/stable/reference/generated/numpy.outer.html#numpy.outer
    """
    if not type(a) is type(b):
        raise TypeError(f"Operation 'outer' requires both arrays be in the same Galois field, not {type(a)} and {type(b)}.")

    if type(a).is_prime_field:
        return _lapack_linalg(a, b, np.outer, out=out, n_sum=1)
    else:
        return np.multiply.outer(a.ravel(), b.ravel(), out=out)


###############################################################################
# Matrix decomposition routines
###############################################################################

def row_reduce(A, ncols=None):
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

    return A_rre


def lu_decompose(A):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise ValueError(f"Argument `A` must be a square matrix, not {A.shape}.")

    field = type(A)
    n = A.shape[0]
    Ai = A.copy()
    L = field.Identity(n)

    for i in range(0, n-1):
        if Ai[i,i] == 0:
            raise ValueError("The LU decomposition of `A` does not exist. Use the LUP decomposition instead.")
        l = Ai[i+1:,i] / Ai[i,i]
        Ai[i+1:,:] -= np.multiply.outer(l, Ai[i,:])
        L[i+1:,i] = l
    U = Ai

    return L, U


def lup_decompose(A):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise ValueError(f"Argument `A` must be a square matrix, not {A.shape}.")

    field = type(A)
    n = A.shape[0]
    Ai = A.copy()
    L = field.Zeros((n,n))
    P = field.Identity(n)

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

        l = Ai[i+1:,i] / Ai[i,i]
        Ai[i+1:,:] -= np.multiply.outer(l, Ai[i,:])  # Zero out rows below row `i`
        L[i,i] = 1  # Set 1 on the diagonal
        L[i+1:,i] = l

    L[-1,-1] = 1  # Set the final diagonal to 1
    U = Ai

    return L, U, P


###############################################################################
# Matrix inversions, solutions, rank, etc
###############################################################################

def matrix_rank(A):
    A_rre = row_reduce(A)
    return np.sum(~np.all(A_rre == 0, axis=1))


def inv(A):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
    field = type(A)
    n = A.shape[0]
    I = field.Identity(n, dtype=A.dtype)

    # Concatenate A and I to get the matrix AI = [A | I]
    AI = np.concatenate((A, I), axis=-1)

    # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
    AI_rre = row_reduce(AI, ncols=n)

    # The rank is the number of non-zero rows of the row reduced echelon form
    rank = np.sum(~np.all(AI_rre[:,0:n] == 0, axis=1))
    if not rank == n:
        raise np.linalg.LinAlgError(f"Argument `A` is singular and not invertible because it does not have full rank of {n}, but rank of {rank}.")

    A_inv = AI_rre[:,-n:]

    return A_inv


def triangular_det(A):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
    idxs = np.arange(0, A.shape[0])
    return np.multiply.reduce(A[idxs,idxs])


def det(A):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
    n = A.shape[0]

    if n == 2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    elif n == 3:
        return A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0])
    else:
        L, U, P = lup_decompose(A)
        idxs = np.arange(0, n)
        nrows = n - np.count_nonzero(P[idxs,idxs]) # The number of moved rows
        S = max(nrows - 1, 0)  # The number of row swaps
        return (-1)**S * triangular_det(L) * triangular_det(U)


def solve(A, b):
    if not type(A) is type(b):
        raise TypeError(f"Arguments `A` and `b` must be of the same Galois field array class, not {type(A)} and {type(b)}.")
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise np.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
    if not b.ndim in [1, 2]:
        raise ValueError(f"Argument `b` must have dimension equal to A or one less, not {b.ndim}.")
    if not A.shape[-1] == b.shape[0]:
        raise ValueError(f"The last dimension of `A` must equal the first dimension of `b`, not {A.shape} and {b.shape}.")

    A_inv = inv(A)
    x = A_inv @ b

    return x
