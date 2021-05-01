"""
A pytest module to test linear algebra routines over Galois fields.
"""
import pytest
import numpy as np

import galois


def test_dot_scalar(field):
    a = field.Random((3,3))
    b = field.Random()
    c = np.dot(a, b)
    assert type(c) is field
    assert np.array_equal(c, a * b)


def test_dot_vector_vector(field):
    a = field.Random(3)
    b = field.Random(3)
    c = np.dot(a, b)
    assert type(c) is field
    assert np.array_equal(c, np.sum(a * b))


def test_dot_matrix_matrix(field):
    A = field.Random((3,3))
    B = field.Random((3,3))
    C = np.dot(A, B)
    assert type(C) is field
    assert np.array_equal(C, A @ B)


def test_dot_tensor_vector(field):
    A = field.Random((3,4,5))
    b = field.Random(5)
    C = np.dot(A, b)
    assert type(C) is field
    assert np.array_equal(C, np.sum(A * b, axis=-1))


def test_matmul_scalar(field):
    A = field.Random((3,3))
    B = field.Random()
    with pytest.raises(ValueError):
        A @ B
    with pytest.raises(ValueError):
        np.matmul(A, B)


def test_matmul_1d_1d(field):
    A = field.Random(3)
    B = field.Random(3)
    C = A @ B
    assert C == np.sum(A * B)
    assert C.shape == ()
    assert type(C) is field
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_2d_1d(field):
    A = field.Random((3,4))
    B = field.Random(4)
    C = A @ B
    assert C[0] == np.sum(A[0,:] * B)  # Spot check
    assert C.shape == (3,)
    assert type(C) is field
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_1d_2d(field):
    A = field.Random(4)
    B = field.Random((4,3))
    C = A @ B
    assert C[0] == np.sum(A * B[:,0])  # Spot check
    assert C.shape == (3,)
    assert type(C) is field
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_2d_2d(field):
    A = field.Random((3,4))
    B = field.Random((4,3))
    C = A @ B
    assert C[0,0] == np.sum(A[0,:] * B[:,0])  # Spot check
    assert C.shape == (3,3)
    assert type(C) is field
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_nd_2d(field):
    A = field.Random((2,3,4))
    B = field.Random((4,3))
    C = A @ B
    assert np.array_equal(C[0,0,0], np.sum(A[0,0,:] * B[:,0]))  # Spot check
    assert C.shape == (2,3,3)
    assert type(C) is field
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_nd_nd(field):
    A = field.Random((2,3,4))
    B = field.Random((2,4,3))
    C = A @ B
    assert np.array_equal(C[0,0,0], np.sum(A[0,0,:] * B[0,:,0]))  # Spot check
    assert C.shape == (2,3,3)
    assert type(C) is field
    assert np.array_equal(A @ B, np.matmul(A, B))


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
        assert np.array_equal(L @ U, A)


def test_lup_decomposition():
    GF = galois.GF(3)
    I = GF.Eye(3)
    for trial in range(100):
        A = GF.Random((3,3))
        L, U, P = A.lup_decompose()
        assert np.array_equal(L @ U, P @ A)


def test_det_2x2(field):
    a = field.Random()
    b = field.Random()
    c = field.Random()
    d = field.Random()
    A = field([[a, b], [c, d]])
    assert np.linalg.det(A) == a*d - b*c


def test_det_3x3(field):
    a = field.Random()
    b = field.Random()
    c = field.Random()
    d = field.Random()
    e = field.Random()
    f = field.Random()
    g = field.Random()
    h = field.Random()
    i = field.Random()
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
    A = full_rank_matrix(field, 3)
    b = field.Random(3)
    x = np.linalg.solve(A, b)
    assert np.array_equal(A @ x, b)
    assert x.shape == b.shape
    assert type(x) is field


def test_solve_2d_2d(field):
    A = full_rank_matrix(field, 3)
    b = field.Random((3,5))
    x = np.linalg.solve(A, b)
    assert np.array_equal(A @ x, b)
    assert x.shape == b.shape
    assert type(x) is field


def full_rank_matrix(field, n):
    A = field.Eye(n)
    while True:
        A = field.Random((n,n))
        if np.linalg.matrix_rank(A) == n:
            break
    return A
