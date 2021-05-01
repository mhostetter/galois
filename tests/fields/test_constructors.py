"""
A pytest module to test instantiation of new Galois field arrays through alternate constructors.
"""
import random

import pytest
import numpy as np

import galois

DTYPES = galois.dtypes.DTYPES + [np.object_]


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_zeros(field, shape):
    a = field.Zeros(shape)
    assert np.all(a == 0)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_zeros_valid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Zeros(shape, dtype=dtype)
    assert np.all(a == 0)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_zeros_invalid_dtype(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        a = field.Zeros(shape, dtype=dtype)


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_ones(field, shape):
    a = field.Ones(shape)
    assert np.all(a == 1)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_ones_valid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Ones(shape, dtype=dtype)
    assert np.all(a == 1)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_ones_invalid_dtype(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        a = field.Ones(shape, dtype=dtype)


def test_eye(field):
    size = 4
    a = field.Identity(size)
    for i in range(size):
        for j in range(size):
            assert a[i,j] == 1 if i == j else a[i,j] == 0
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == (size,size)


def test_eye_valid_dtype(field):
    dtype = valid_dtype(field)
    size = 4
    a = field.Identity(size, dtype=dtype)
    for i in range(size):
        for j in range(size):
            assert a[i,j] == 1 if i == j else a[i,j] == 0
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == (size,size)


def test_eye_invalid_dtype(field):
    dtype = invalid_dtype(field)
    size = 4
    with pytest.raises(TypeError):
        a = field.Identity(size, dtype=dtype)


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_random(field, shape):
    a = field.Random(shape)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_random_valid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Random(shape, dtype=dtype)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_random_invalid_dtype(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        a = field.Random(shape, dtype=dtype)


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_vector_valid_dtype(field, shape):
    v_dtype = valid_dtype(field.prime_subfield)
    v_shape = list(shape) + [field.degree]
    v = field.prime_subfield.Random(v_shape, dtype=v_dtype)
    dtype = valid_dtype(field)
    a = field.Vector(v, dtype=dtype)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_vector_invalid_dtype(field, shape):
    v_dtype = valid_dtype(field.prime_subfield)
    v_shape = list(shape) + [field.degree]
    v = field.prime_subfield.Random(v_shape, dtype=v_dtype)
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        a = field.Vector(v, dtype=dtype)


def valid_dtype(field):
    return random.choice(field.dtypes)


def invalid_dtype(field):
    return random.choice([dtype for dtype in DTYPES if dtype not in field.dtypes])
