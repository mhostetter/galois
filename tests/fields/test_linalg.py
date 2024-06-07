"""
A pytest module to test linear algebra routines over Galois fields.
"""

import random

import numpy as np
import pytest

import galois

from .conftest import array_equal


def test_dot_exceptions():
    with pytest.raises(TypeError):
        a = galois.GF(2**4).Random(5)
        b = galois.GF(2**5).Random(5)
        np.dot(a, b)


def test_dot_scalar(field):
    dtype = random.choice(field.dtypes)
    a = field.Random((3, 3), dtype=dtype)
    b = field.Random(dtype=dtype)

    c = np.dot(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, a * b)

    c = a.dot(b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, a * b)


def test_dot_vector_vector(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(3, dtype=dtype)
    b = field.Random(3, dtype=dtype)

    c = np.dot(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.sum(a * b))

    c = a.dot(b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.sum(a * b))


def test_dot_matrix_matrix(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3, 3), dtype=dtype)
    B = field.Random((3, 3), dtype=dtype)

    C = np.dot(A, B)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, A @ B)

    C = A.dot(B)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, A @ B)


def test_dot_tensor_vector(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3, 4, 5), dtype=dtype)
    b = field.Random(5, dtype=dtype)

    C = np.dot(A, b)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, np.sum(A * b, axis=-1))

    C = A.dot(b)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, np.sum(A * b, axis=-1))


def test_vdot_exceptions():
    with pytest.raises(TypeError):
        a = galois.GF(2**4).Random(5)
        b = galois.GF(2**5).Random(5)
        np.vdot(a, b)


def test_vdot_scalar(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(dtype=dtype)
    b = field.Random(dtype=dtype)
    c = np.vdot(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, a * b)


def test_vdot_vector_vector(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(3, dtype=dtype)
    b = field.Random(3, dtype=dtype)
    c = np.vdot(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.sum(a * b))


def test_vdot_matrix_matrix(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3, 3), dtype=dtype)
    B = field.Random((3, 3), dtype=dtype)
    C = np.vdot(A, B)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, np.sum(A.flatten() * B.flatten()))


def test_inner_exceptions():
    with pytest.raises(TypeError):
        a = galois.GF(2**4).Random(5)
        b = galois.GF(2**5).Random(5)
        np.inner(a, b)
    with pytest.raises(ValueError):
        a = galois.GF(2**4).Random((3, 4))
        b = galois.GF(2**4).Random((3, 5))
        np.inner(a, b)


def test_inner_scalar_scalar(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(dtype=dtype)
    b = field.Random(dtype=dtype)
    c = np.inner(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert c == a * b


def test_inner_vector_vector(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(3, dtype=dtype)
    b = field.Random(3, dtype=dtype)
    c = np.inner(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.sum(a * b))


def test_outer_exceptions():
    with pytest.raises(TypeError):
        a = galois.GF(2**4).Random(5)
        b = galois.GF(2**5).Random(5)
        np.outer(a, b)


def test_outer_vector_vector(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(3, dtype=dtype)
    b = field.Random(4, dtype=dtype)
    c = np.outer(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.multiply.outer(a, b))


def test_outer_nd_nd(field):
    dtype = random.choice(field.dtypes)
    a = field.Random((3, 3), dtype=dtype)
    b = field.Random((4, 4), dtype=dtype)
    c = np.outer(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.multiply.outer(a.ravel(), b.ravel()))


def test_matmul_scalar(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3, 3), dtype=dtype)
    B = field.Random(dtype=dtype)
    with pytest.raises(ValueError):
        A @ B
    with pytest.raises(ValueError):
        np.matmul(A, B)


def test_matmul_1d_1d(field):
    dtype = random.choice(field.dtypes)
    A = field.Random(3, dtype=dtype)
    B = field.Random(3, dtype=dtype)
    C = A @ B
    assert C == np.sum(A * B)
    assert C.shape == ()
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(A @ B, np.matmul(A, B))


def test_matmul_2d_1d(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3, 4), dtype=dtype)
    B = field.Random(4, dtype=dtype)
    C = A @ B
    assert C[0] == np.sum(A[0, :] * B)  # Spot check
    assert C.shape == (3,)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(A @ B, np.matmul(A, B))


def test_matmul_1d_2d(field):
    dtype = random.choice(field.dtypes)
    A = field.Random(4, dtype=dtype)
    B = field.Random((4, 3), dtype=dtype)
    C = A @ B
    assert C[0] == np.sum(A * B[:, 0])  # Spot check
    assert C.shape == (3,)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(A @ B, np.matmul(A, B))


def test_matmul_2d_2d(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3, 4), dtype=dtype)
    B = field.Random((4, 3), dtype=dtype)
    C = A @ B
    assert C[0, 0] == np.sum(A[0, :] * B[:, 0])  # Spot check
    assert C.shape == (3, 3)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(A @ B, np.matmul(A, B))


# def test_matmul_nd_2d(field):
#     A = field.Random((2,3,4), dtype=dtype)
#     B = field.Random((4,3), dtype=dtype)
#     C = A @ B
#     assert array_equal(C[0,0,0], np.sum(A[0,0,:] * B[:,0]))  # Spot check
#     assert C.shape == (2,3,3)
#     assert type(C) is field
#     assert array_equal(A @ B, np.matmul(A, B))


# def test_matmul_nd_nd(field):
#     A = field.Random((2,3,4), dtype=dtype)
#     B = field.Random((2,4,3), dtype=dtype)
#     C = A @ B
#     assert array_equal(C[0,0,0], np.sum(A[0,0,:] * B[0,:,0]))  # Spot check
#     assert C.shape == (2,3,3)
#     assert type(C) is field
#     assert array_equal(A @ B, np.matmul(A, B))


def full_rank_matrix(field, n, dtype):
    A = field.Identity(n, dtype=dtype)
    while True:
        A = field.Random((n, n), dtype=dtype)
        if np.linalg.matrix_rank(A) == n:
            break
    return A


###############################################################################
# Tests against Sage test vectors
###############################################################################


def test_matrix_multiply(field_matrix_multiply):
    GF, X, Y, Z = (
        field_matrix_multiply["GF"],
        field_matrix_multiply["X"],
        field_matrix_multiply["Y"],
        field_matrix_multiply["Z"],
    )
    for x, y, z_truth in zip(X, Y, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        y = y.astype(dtype)
        z = x @ y
        assert np.array_equal(z, z_truth)
        assert type(z) is GF


def test_row_reduce_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.row_reduce()
    with pytest.raises(ValueError):
        A = GF.Random((2, 2, 2))
        A.row_reduce()


def test_row_reduce(field_row_reduce):
    GF, X, Z = field_row_reduce["GF"], field_row_reduce["X"], field_row_reduce["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.row_reduce()
        assert np.array_equal(z, z_truth)
        assert type(z) is GF


def test_row_reduce_eye_right():
    GF = galois.GF(2)
    H = GF([[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
    H_rre = H.row_reduce(eye="right")
    assert type(H_rre) is GF
    assert np.array_equal(
        H_rre, [[0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 1]]
    )


def test_lu_decompose_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.lu_decompose()
    with pytest.raises(ValueError):
        A = GF.Random((2, 2, 2))
        A.lu_decompose()


def test_lu_decompose(field_lu_decompose):
    GF, X, L, U = field_lu_decompose["GF"], field_lu_decompose["X"], field_lu_decompose["L"], field_lu_decompose["U"]
    for x, l_truth, u_truth in zip(X, L, U):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        l, u = x.lu_decompose()
        assert np.array_equal(l, l_truth)
        assert np.array_equal(u, u_truth)
        assert type(l) is GF
        assert type(u) is GF


def test_plu_decompose_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.plu_decompose()
    with pytest.raises(ValueError):
        A = GF.Random((2, 2, 2))
        A.plu_decompose()


def test_plu_decompose(field_plu_decompose):
    GF, X, P, L, U = (
        field_plu_decompose["GF"],
        field_plu_decompose["X"],
        field_plu_decompose["P"],
        field_plu_decompose["L"],
        field_plu_decompose["U"],
    )
    for x, p_truth, l_truth, u_truth in zip(X, P, L, U):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        p, l, u = x.plu_decompose()
        assert np.array_equal(p, p_truth)
        assert np.array_equal(l, l_truth)
        assert np.array_equal(u, u_truth)
        assert type(p) is GF
        assert type(l) is GF
        assert type(u) is GF


def test_bug_476():
    """
    See https://github.com/mhostetter/galois/issues/476.
    """
    GF = galois.GF(2)
    A = GF(
        [
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    P, L, U = A.plu_decompose()
    PLU = P @ L @ U
    assert np.array_equal(A, PLU)


def test_matrix_inverse_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random(5)
        np.linalg.inv(A)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random((2, 2, 2))
        np.linalg.inv(A)


def test_matrix_inverse(field_matrix_inverse):
    GF, X, Z = field_matrix_inverse["GF"], field_matrix_inverse["X"], field_matrix_inverse["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = np.linalg.inv(x)
        assert np.array_equal(z, z_truth)
        assert type(z) is GF


def test_matrix_determinant_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random(5)
        np.linalg.det(A)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random((2, 2, 2))
        np.linalg.det(A)


def test_matrix_determinant(field_matrix_determinant):
    GF, X, Z = field_matrix_determinant["GF"], field_matrix_determinant["X"], field_matrix_determinant["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = np.linalg.det(x)
        assert z == z_truth
        assert type(z) is GF


def test_matrix_solve_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(TypeError):
        A = galois.GF(2**4).Random((3, 3))
        b = galois.GF(2**5).Random(3)
        np.linalg.solve(A, b)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random((2, 3))
        b = GF.Random(3)
        np.linalg.solve(A, b)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random((2, 2))
        b = GF.Random((2, 2, 2))
        np.linalg.solve(A, b)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random((2, 2))
        b = GF.Random()
        np.linalg.solve(A, b)
    with pytest.raises(np.linalg.LinAlgError):
        A = GF.Random((2, 2))
        b = GF.Random(3)
        np.linalg.solve(A, b)


def test_matrix_solve(field_matrix_solve):
    GF, X, Y, Z = field_matrix_solve["GF"], field_matrix_solve["X"], field_matrix_solve["Y"], field_matrix_solve["Z"]
    # np.linalg.solve(x, y) = z corresponds to x @ z = y
    for x, y, z_truth in zip(X, Y, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        y = y.astype(dtype)
        z = np.linalg.solve(x, y)
        assert np.array_equal(z, z_truth)
        assert type(z) is GF


def test_row_space_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.row_space()
    with pytest.raises(ValueError):
        A = GF.Random((2, 2, 2))
        A.row_space()


def test_row_space(field_row_space):
    GF, X, Z = field_row_space["GF"], field_row_space["X"], field_row_space["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.row_space()
        assert np.array_equal(z, z_truth)
        assert type(z) is GF


def test_column_space_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.column_space()
    with pytest.raises(ValueError):
        A = GF.Random((2, 2, 2))
        A.column_space()


def test_column_space(field_column_space):
    GF, X, Z = field_column_space["GF"], field_column_space["X"], field_column_space["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.column_space()
        assert np.array_equal(z, z_truth)
        assert type(z) is GF


def test_left_null_space_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.left_null_space()
    with pytest.raises(ValueError):
        A = GF.Random((2, 2, 2))
        A.left_null_space()


def test_left_null_space(field_left_null_space):
    GF, X, Z = field_left_null_space["GF"], field_left_null_space["X"], field_left_null_space["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.left_null_space()
        assert np.array_equal(z, z_truth)
        assert type(z) is GF


def test_null_space_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.null_space()
    with pytest.raises(ValueError):
        A = GF.Random((2, 2, 2))
        A.null_space()


def test_null_space(field_null_space):
    GF, X, Z = field_null_space["GF"], field_null_space["X"], field_null_space["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.null_space()
        assert np.array_equal(z, z_truth)
        assert type(z) is GF
