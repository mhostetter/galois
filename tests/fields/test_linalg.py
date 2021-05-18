"""
A pytest module to test linear algebra routines over Galois fields.
"""
import random

import pytest
import numpy as np

import galois

from ..helper import array_equal


def test_dot_scalar(field):
    dtype = random.choice(field.dtypes)
    a = field.Random((3,3), dtype=dtype)
    b = field.Random(dtype=dtype)
    c = np.dot(a, b)
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


def test_dot_matrix_matrix(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3,3), dtype=dtype)
    B = field.Random((3,3), dtype=dtype)
    C = np.dot(A, B)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, A @ B)


def test_dot_tensor_vector(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3,4,5), dtype=dtype)
    b = field.Random(5, dtype=dtype)
    C = np.dot(A, b)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, np.sum(A * b, axis=-1))


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
    A = field.Random((3,3), dtype=dtype)
    B = field.Random((3,3), dtype=dtype)
    C = np.vdot(A, B)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(C, np.sum(A.flatten() * B.flatten()))


def test_inner_scalar_scalar(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(dtype=dtype)
    b = field.Random(dtype=dtype)
    c = np.inner(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert c == a*b


def test_inner_vector_vector(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(3, dtype=dtype)
    b = field.Random(3, dtype=dtype)
    c = np.inner(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.sum(a * b))


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
    a = field.Random((3,3), dtype=dtype)
    b = field.Random((4,4), dtype=dtype)
    c = np.outer(a, b)
    assert type(c) is field
    assert c.dtype == dtype
    assert array_equal(c, np.multiply.outer(a.ravel(), b.ravel()))


def test_matmul_scalar(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3,3), dtype=dtype)
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
    A = field.Random((3,4), dtype=dtype)
    B = field.Random(4, dtype=dtype)
    C = A @ B
    assert C[0] == np.sum(A[0,:] * B)  # Spot check
    assert C.shape == (3,)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(A @ B, np.matmul(A, B))


def test_matmul_1d_2d(field):
    dtype = random.choice(field.dtypes)
    A = field.Random(4, dtype=dtype)
    B = field.Random((4,3), dtype=dtype)
    C = A @ B
    assert C[0] == np.sum(A * B[:,0])  # Spot check
    assert C.shape == (3,)
    assert type(C) is field
    assert C.dtype == dtype
    assert array_equal(A @ B, np.matmul(A, B))


def test_matmul_2d_2d(field):
    dtype = random.choice(field.dtypes)
    A = field.Random((3,4), dtype=dtype)
    B = field.Random((4,3), dtype=dtype)
    C = A @ B
    assert C[0,0] == np.sum(A[0,:] * B[:,0])  # Spot check
    assert C.shape == (3,3)
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


def test_lu_decomposition():
    GF = galois.GF(3)
    for trial in range(100):
        A = GF.Random((3,3))
        try:
            L, U = A.lu_decompose()
        except ValueError:
            # The LU decomposition isn't guaranteed to exist
            continue
        # For those decompositions that exist, confirm they are correct
        assert array_equal(L @ U, A)


def test_lup_decomposition():
    GF = galois.GF(3)
    I = GF.Identity(3)
    for trial in range(100):
        A = GF.Random((3,3))
        L, U, P = A.lup_decompose()
        assert array_equal(L @ U, P @ A)


def test_det_2x2(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(dtype=dtype)
    b = field.Random(dtype=dtype)
    c = field.Random(dtype=dtype)
    d = field.Random(dtype=dtype)
    A = field([[a, b], [c, d]])
    assert np.linalg.det(A) == a*d - b*c


def test_det_3x3(field):
    dtype = random.choice(field.dtypes)
    a = field.Random(dtype=dtype)
    b = field.Random(dtype=dtype)
    c = field.Random(dtype=dtype)
    d = field.Random(dtype=dtype)
    e = field.Random(dtype=dtype)
    f = field.Random(dtype=dtype)
    g = field.Random(dtype=dtype)
    h = field.Random(dtype=dtype)
    i = field.Random(dtype=dtype)
    A = field([[a, b, c], [d, e, f], [g, h, i]])
    assert np.linalg.det(A) == a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h


def test_det_3x3_repeated():
    GF = galois.GF(3)
    for trial in range(100):
        a = GF.Random()
        b = GF.Random()
        c = GF.Random()
        d = GF.Random()
        e = GF.Random()
        f = GF.Random()
        g = GF.Random()
        h = GF.Random()
        i = GF.Random()
        A = GF([[a, b, c], [d, e, f], [g, h, i]])
        assert np.linalg.det(A) == a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h


def test_solve_2d_1d(field):
    dtype = random.choice(field.dtypes)
    A = full_rank_matrix(field, 3, dtype)
    b = field.Random(3, dtype=dtype)
    x = np.linalg.solve(A, b)
    assert type(x) is field
    assert x.dtype == dtype
    assert array_equal(A @ x, b)
    assert x.shape == b.shape


def test_solve_2d_2d(field):
    dtype = random.choice(field.dtypes)
    A = full_rank_matrix(field, 3, dtype)
    b = field.Random((3,5), dtype=dtype)
    x = np.linalg.solve(A, b)
    assert type(x) is field
    assert x.dtype == dtype
    assert array_equal(A @ x, b)
    assert x.shape == b.shape


def full_rank_matrix(field, n, dtype):
    A = field.Identity(n, dtype=dtype)
    while True:
        A = field.Random((n,n), dtype=dtype)
        if np.linalg.matrix_rank(A) == n:
            break
    return A
