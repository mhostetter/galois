import numpy as np


def matmul(A, B, **kwargs):
    # pylint: disable=too-many-branches
    if not type(A) is type(B):
        raise TypeError(f"Operation 'matmul' requires both arrays be in the same Galois field, not {type(A)} and {type(B)}.")
    if not (A.ndim >= 1 and B.ndim >= 1):
        raise ValueError(f"Operation 'matmul' requires both arrays have dimension at least 1, not {A.ndim}-D and {B.ndim}-D.")

    prepend, append = False, False
    if A.ndim == 1:
        A = A.reshape((1,A.size))
        prepend = True
    if B.ndim == 1:
        B = B.reshape((B.size,1))
        append = True

    if not A.shape[-1] == B.shape[-2]:
        raise ValueError(f"Operation 'matmul' requires the last dimension of A to match the second-to-last dimension of B, not {A.shape} and {B.shape}.")

    if A.ndim > 2 and B.ndim == 2:
        new_shape = list(A.shape[:-2]) + list(B.shape)
        B = np.broadcast_to(B, new_shape)
    if B.ndim > 2 and A.ndim == 2:
        new_shape = list(B.shape[:-2]) + list(A.shape)
        A = np.broadcast_to(A, new_shape)

    out_shape = tuple(list(A.shape[:-2]) + [A.shape[-2]] + [B.shape[-1]])
    if "out" in kwargs:
        assert len(kwargs["out"]) == 1
        C = kwargs["out"][0]
        if not C.shape == out_shape:
            raise ValueError(f"Argument `out` needs to have shape {out_shape}, not {C.shape}.")
    else:
        C = type(A).Zeros(out_shape)

    for i in range(A.shape[-2]):
        for j in range(B.shape[-1]):
            C[...,i,j] = np.sum(A[...,i,:] * B[...,:,j], axis=-1)

    shape = list(C.shape)
    if prepend:
        shape = shape[1:]
    if append:
        shape = shape[:-1]
    C = C.reshape(shape)

    return C


def row_reduce(A, ncols=None):
    if not A.ndim == 2:
        raise ValueError(f"Only 2-D matrices can be converted to reduced row echelon form, not {A.ndim}-D.")

    ncols = A.shape[1] if ncols is None else ncols
    A_rre = np.copy(A)
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
    Ai = np.copy(A)
    L = field.Eye(n)

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
    Ai = np.copy(A)
    L = field.Zeros((n,n))
    P = field.Eye(n)

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


def matrix_rank(A, **kwargs):  # pylint: disable=unused-argument
    A_rre = row_reduce(A)
    return np.sum(~np.all(A_rre == 0, axis=1))


def inv(A):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise ValueError(f"Argument `A` must be square, not {A.shape}.")
    field = type(A)
    n = A.shape[0]
    I = field.Eye(n)

    # Concatenate A and I to get the matrix AI = [A | I]
    AI = np.concatenate((A, I), axis=-1)

    # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
    AI_rre = row_reduce(AI, ncols=n)

    # The rank is the number of non-zero rows of the row reduced echelon form
    rank = np.sum(~np.all(AI_rre[:,0:n] == 0, axis=1))
    if not rank == n:
        raise ValueError(f"The matrix `A` is not invertible because it does not have full rank of {n}, but rank {rank}.")

    A_inv = AI_rre[:,-n:]

    return A_inv


def triangular_det(A):
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    idxs = np.arange(0, A.shape[0])
    return np.multiply.reduce(A[idxs,idxs])


def det(A):
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    if n == 2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    else:
        L, U, P = lup_decompose(A)
        idxs = np.arange(0, n)
        nrows = n - np.count_nonzero(P[idxs,idxs]) # The number of moved rows
        S = max(nrows - 1, 0)  # The number of row swaps
        return (-1)**S * triangular_det(L) * triangular_det(U)


def solve(A, b, **kwargs):  # pylint: disable=unused-argument
    if not type(A) is type(b):
        raise TypeError(f"Arguments `A` and `b` must be of the same Galois field array class, not {type(A)} and {type(b)}.")
    # if not A.ndim >= 2 and A.shape[-2] == A.shape[-1]:
    #     raise ValueError(f"Argument `A` must be square in the last two dimensions, not {A.shape}.")
    if not A.ndim == 2 and A.shape[-2] == A.shape[-1]:
        raise ValueError(f"Argument `A` must be square, not {A.shape}.")
    if not b.ndim in [A.ndim - 1, A.ndim]:
        raise ValueError(f"Argument `b` must have dimension equal to A or one less, not {b.ndim}.")
    if not A.shape[-1] == b.shape[0]:
        raise ValueError(f"The last dimension of `A` must equal the first dimension of `b`, not {A.shape} and {b.shape}.")

    A_inv = inv(A)
    x = A_inv @ b

    return x
