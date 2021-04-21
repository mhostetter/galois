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
