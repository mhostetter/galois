"""
A pytest module to test linear algebra routines over Galois fields.
"""
import pytest
import numpy as np

import galois


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
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_2d_1d(field):
    A = field.Random((3,4))
    B = field.Random(4)
    C = A @ B
    assert C[0] == np.sum(A[0,:] * B)  # Spot check
    assert C.shape == (3,)
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_1d_2d(field):
    A = field.Random(4)
    B = field.Random((4,3))
    C = A @ B
    assert C[0] == np.sum(A * B[:,0])  # Spot check
    assert C.shape == (3,)
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_2d_2d(field):
    A = field.Random((3,4))
    B = field.Random((4,3))
    C = A @ B
    assert C[0,0] == np.sum(A[0,:] * B[:,0])  # Spot check
    assert C.shape == (3,3)
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_nd_2d(field):
    A = field.Random((2,3,4))
    B = field.Random((4,3))
    C = A @ B
    assert np.array_equal(C[0,0,0], np.sum(A[0,0,:] * B[:,0]))  # Spot check
    assert C.shape == (2,3,3)
    assert np.array_equal(A @ B, np.matmul(A, B))


def test_matmul_nd_nd(field):
    A = field.Random((2,3,4))
    B = field.Random((2,4,3))
    C = A @ B
    assert np.array_equal(C[0,0,0], np.sum(A[0,0,:] * B[0,:,0]))  # Spot check
    assert C.shape == (2,3,3)
    assert np.array_equal(A @ B, np.matmul(A, B))
