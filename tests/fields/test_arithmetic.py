"""
A pytest module to test the accuracy of FieldArray arithmetic.
"""

import random

import numpy as np
import pytest

import galois

from .conftest import randint

# TODO: Add scalar arithmetic and array/scalar and radd, etc


def test_add(field_add):
    GF, X, Y, Z = field_add["GF"], field_add["X"], field_add["Y"], field_add["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y.astype(dtype)

    z = x + y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_additive_inverse(field_additive_inverse):
    GF, X, Z = field_additive_inverse["GF"], field_additive_inverse["X"], field_additive_inverse["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)

    z = -x
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_subtract(field_subtract):
    GF, X, Y, Z = field_subtract["GF"], field_subtract["X"], field_subtract["Y"], field_subtract["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y.astype(dtype)

    z = x - y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_multiply(field_multiply):
    GF, X, Y, Z = field_multiply["GF"], field_multiply["X"], field_multiply["Y"], field_multiply["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y.astype(dtype)

    z = x * y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_scalar_multiply(field_scalar_multiply):
    GF, X, Y, Z = (
        field_scalar_multiply["GF"],
        field_scalar_multiply["X"],
        field_scalar_multiply["Y"],
        field_scalar_multiply["Z"],
    )
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


def test_multiplicative_inverse(field_multiplicative_inverse):
    GF, X, Z = field_multiplicative_inverse["GF"], field_multiplicative_inverse["X"], field_multiplicative_inverse["Z"]
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

    z = x**-1
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_divide(field_divide):
    GF, X, Y, Z = field_divide["GF"], field_divide["X"], field_divide["Y"], field_divide["Z"]
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


def test_divmod(field_divide):
    GF = field_divide["GF"]
    dtype = random.choice(GF.dtypes)
    x = GF.Random(10, dtype=dtype)
    y = GF.Random(10, low=1, dtype=dtype)

    q, r = np.divmod(x, y)
    assert np.array_equal(q, x // y)
    assert type(q) is GF
    assert q.dtype == dtype
    assert np.all(r == 0)
    assert type(r) is GF
    assert r.dtype == dtype


def test_mod(field_divide):
    GF = field_divide["GF"]
    dtype = random.choice(GF.dtypes)
    x = GF.Random(10, dtype=dtype)
    y = GF.Random(10, low=1, dtype=dtype)

    r = x % y
    assert np.all(r == 0)
    assert type(r) is GF
    assert r.dtype == dtype


def test_power(field_power):
    GF, X, Y, Z = field_power["GF"], field_power["X"], field_power["Y"], field_power["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    y = Y  # Don't convert this, it's not a field element

    z = x**y
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_power_zero_to_zero(field_power):
    GF = field_power["GF"]
    dtype = random.choice(GF.dtypes)
    x = GF.Zeros(10, dtype=dtype)
    y = np.zeros(10, GF.dtypes[-1])
    z = x**y
    Z = np.ones(10, GF.dtypes[-1])
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_power_zero_to_positive_integer(field_power):
    GF = field_power["GF"]
    dtype = random.choice(GF.dtypes)
    x = GF.Zeros(10, dtype=dtype)
    y = randint(1, 2 * GF.order, 10, GF.dtypes[-1])
    z = x**y
    Z = np.zeros(10, GF.dtypes[-1])
    assert np.array_equal(z, Z)
    assert type(z) is GF
    assert z.dtype == dtype


def test_square(field_power):
    GF = field_power["GF"]
    dtype = random.choice(GF.dtypes)
    x = GF.Random(10, dtype=dtype)
    x[0:2] = [0, 1]

    z = x**2
    assert np.array_equal(z, x * x)
    assert type(z) is GF
    assert z.dtype == dtype


def test_log(field_log):
    GF, X, Z = field_log["GF"], field_log["X"], field_log["Z"]
    dtype = random.choice(GF.dtypes)
    if GF.order > 2**16:  # TODO: Skip slow log() for very large fields
        return
    x = X.astype(dtype)
    z = np.log(x)
    assert np.array_equal(z, Z)
    z = x.log()
    assert np.array_equal(z, Z)


def test_log_different_base(field_log):
    GF, X = field_log["GF"], field_log["X"]
    dtype = random.choice(GF.dtypes)
    if GF.order > 2**16:  # TODO: Skip slow log() for very large fields
        return
    x = X.astype(dtype)
    beta = GF.primitive_elements[-1]
    z = x.log(beta)
    assert np.array_equal(beta**z, x)


@pytest.mark.parametrize("ufunc_mode", ["jit-calculate", "python-calculate"])
def test_log_pollard_rho(ufunc_mode):
    """
    The Pollard-rho discrete logarithm algorithm is only applicable for fields when p^m - 1 is prime.
    """
    GF = galois.GF(2**5, compile=ufunc_mode)
    assert isinstance(GF._log, galois._domains._calculate.log_pollard_rho)
    dtype = random.choice(GF.dtypes)
    x = GF.Random(10, low=1, dtype=dtype)

    alpha = GF.primitive_element
    z = np.log(x)
    assert np.array_equal(alpha**z, x)
    z = x.log()
    assert np.array_equal(alpha**z, x)

    beta = GF.primitive_elements[-1]
    z = x.log(beta)
    assert np.array_equal(beta**z, x)


# TODO: Skip slow log() for very large fields
# def test_log_pollard_rho_python():
#     GF = galois.GF(2**61)
#     assert isinstance(GF._log, galois._domains._calculate.log_pollard_rho)
#     dtype = random.choice(GF.dtypes)
#     x = GF.Random(low=1, dtype=dtype)

#     alpha = GF.primitive_element
#     z = x.log()
#     assert np.array_equal(alpha ** z, x)


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
