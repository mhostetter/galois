"""
A pytest module to test instantiation of new Galois field arrays through alternate constructors.
"""
import random

import pytest
import numpy as np

import galois

DTYPES = galois.array.DTYPES + [np.object_]


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_zeros(field, shape):
    a = field.Zeros(shape)
    assert np.all(a == 0)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_zeros_valid_dtypes(field, shape):
    dtype = valid_dtype(field)
    a = field.Zeros(shape, dtype=dtype)
    assert np.all(a == 0)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_zeros_valid_dtypes(field, shape):
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
def test_ones_valid_dtypes(field, shape):
    dtype = valid_dtype(field)
    a = field.Ones(shape, dtype=dtype)
    assert np.all(a == 1)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_ones_valid_dtypes(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        a = field.Ones(shape, dtype=dtype)


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_random(field, shape):
    a = field.Random(shape)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_random_valid_dtypes(field, shape):
    dtype = valid_dtype(field)
    a = field.Random(shape, dtype=dtype)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4,4)])
def test_random_valid_dtypes(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        a = field.Random(shape, dtype=dtype)


def valid_dtype(field):
    return random.choice(field.dtypes)


def invalid_dtype(field):
    return random.choice([dtype for dtype in DTYPES if dtype not in field.dtypes])
