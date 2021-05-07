"""
A pytest module to test the accuracy of Galois field array arithmetic.
"""
import random

import pytest
import numpy as np

import galois

from ..helper import randint


# TODO: Add scalar arithmetic and array/scalar and radd, etc


def test_add(add):
    GF, X, Y, Z = add["GF"], add["X"], add["Y"], add["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y.astype(dtype)

    z = x + y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_subtract(subtract):
    GF, X, Y, Z = subtract["GF"], subtract["X"], subtract["Y"], subtract["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y.astype(dtype)

    z = x - y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_multiply(multiply):
    GF, X, Y, Z = multiply["GF"], multiply["X"], multiply["Y"], multiply["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y.astype(dtype)

    z = x * y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_divide(divide):
    GF, X, Y, Z = divide["GF"], divide["X"], divide["Y"], divide["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y.astype(dtype)

    z = x / y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype

    z = x // y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_additive_inverse(additive_inverse):
    GF, X, Z = additive_inverse["GF"], additive_inverse["X"], additive_inverse["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)

    z = -x
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_multiplicative_inverse(multiplicative_inverse):
    GF, X, Z = multiplicative_inverse["GF"], multiplicative_inverse["X"], multiplicative_inverse["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)

    z = GF(1, dtype=dtype) / x  # Need dtype of "1" to be as large as x for `z.dtype == dtype`
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype

    z = GF(1, dtype=dtype) // x  # Need dtype of "1" to be as large as x for `z.dtype == dtype`
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype

    z = x ** -1
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_scalar_multiply(scalar_multiply):
    GF, X, Y, Z = scalar_multiply["GF"], scalar_multiply["X"], scalar_multiply["Y"], scalar_multiply["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y  # Don't convert this, it's not a field element

    z = x * y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype

    z = y * x
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_power(power):
    GF, X, Y, Z = power["GF"], power["X"], power["Y"], power["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y  # Don't convert this, it's not a field element

    z = x ** y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_power_zero_to_zero(power):
    GF = power["GF"]
    dtype = random.choice(GF.dtypes)
    x = GF.Zeros(10, dtype=dtype)
    y = np.zeros(10, GF.dtypes[-1])
    z = x ** y
    Z = np.ones(10, GF.dtypes[-1])
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_power_zero_to_positive_integer(power):
    GF = power["GF"]
    dtype = random.choice(GF.dtypes)
    x = GF.Zeros(10, dtype=dtype)
    y = randint(1, 2*GF.order, 10, GF.dtypes[-1])
    z = x ** y
    Z = np.zeros(10, GF.dtypes[-1])
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_square(power):
    GF, X, Y, Z = power["GF"], power["X"], power["Y"], power["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y  # Don't convert this, it's not a field element

    # Not guaranteed to have y=2 for "sparse" LUTs
    if np.where(Y == 2)[1].size > 0:
        j = np.where(y == 2)[1][0]  # Index of Y where y=2
        x = x[:,j]
        z = x ** 2
        assert np.array_equal(z, Z[:,j])
        assert type(z) is GF
        assert z.dtype == dtype


def test_log(log):
    GF, X, Z = log["GF"], log["X"], log["Z"]
    dtype = random.choice(GF.dtypes)
    if GF.order > 2**16:  # TODO: Skip slow log() for very large fields
        return
    x = X.astype(dtype)
    z = np.log(x)
    assert np.array_equal(z, Z)


# class TestArithmeticNonField:


#     def test_scalar_multiply_int_scalar(scalar_multiply):
#         X, Y, Z = scalar_multiply["X"], scalar_multiply["Y"], scalar_multiply["Z"]
#         i = np.random.randint(0, Z.shape[0], 10)  # Random x indices
#         j = np.random.randint(0, Z.shape[1])  # Random y index
#         x = X[i,0]
#         y = Y[0,j]
#         assert np.array_equal(x * y, Z[i,j])
#         assert np.array_equal(y * x, Z[i,j])


#     def test_scalar_multiply_int_array(scalar_multiply):
#         X, Y, Z = scalar_multiply["X"], scalar_multiply["Y"], scalar_multiply["Z"]
#         i = np.random.randint(0, Z.shape[0], 10)  # Random x indices
#         j = np.random.randint(0, Z.shape[1])  # Random y index
#         x = X[i,0]
#         y = Y[0,j]
#         assert np.array_equal(x * y, Z[i,j])
#         assert np.array_equal(y * x, Z[i,j])


#     def test_rmul_int_scalar(scalar_multiply):
#         GF, X, Y, Z = scalar_multiply["GF"], scalar_multiply["X"], scalar_multiply["Y"], scalar_multiply["Z"]
#         i = np.random.randint(0, Z.shape[0])  # Random x index
#         j = np.random.randint(0, Z.shape[1])  # Random y index

#         x = X.copy()
#         y = Y[i,j]  # Integer, non-field element
#         x[i,j] *= y
#         assert x[i,j] == Z[i,j]
#         assert isinstance(x, GF)

#         # TODO: Should this work?
#         # x = X
#         # y = Y[i,j].copy()  # Integer, non-field element
#         # y *= x[i,j]
#         # assert y == Z[i,j]
#         # assert isinstance(y, GF)


#     def test_rmul_int_array(scalar_multiply):
#         GF, X, Y, Z = scalar_multiply["GF"], scalar_multiply["X"], scalar_multiply["Y"], scalar_multiply["Z"]
#         i = np.random.randint(0, Z.shape[0])  # Random x index

#         x = X.copy()
#         y = Y[i,:]
#         x[i,:] *= y
#         assert np.array_equal(x[i,:], Z[i,:])
#         assert isinstance(x, GF)

#         x = X
#         y = Y[i,:].copy()
#         y *= x[i,:]
#         assert np.array_equal(y, Z[i,:])
#         assert isinstance(y, GF)
