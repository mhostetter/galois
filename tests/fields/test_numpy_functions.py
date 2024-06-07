"""
A pytest module to test NumPy methods, both supported and unsupported.
Numpy methods are selected from this API reference:
https://numpy.org/doc/stable/reference/routines.array-manipulation.html
"""

import random

import numpy as np
import pytest

from .conftest import array_equal, randint

###############################################################################
# Basic operations
###############################################################################


def test_copy(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 3)
    a = field.Random(shape, dtype=dtype)
    b = np.copy(a)
    assert type(b) is np.ndarray
    assert b is not a

    d = a.copy()
    assert type(d) is field
    assert d is not a


def test_shape(field):
    dtype = random.choice(field.dtypes)
    shape = ()
    a = field.Random(shape, dtype=dtype)
    assert a.shape == shape
    assert np.shape(a) == shape

    shape = (3,)
    a = field.Random(shape, dtype=dtype)
    assert a.shape == shape
    assert np.shape(a) == shape

    shape = (3, 4, 5)
    a = field.Random(shape, dtype=dtype)
    assert a.shape == shape
    assert np.shape(a) == shape


###############################################################################
# Changing array shape
###############################################################################


def test_reshape(field):
    dtype = random.choice(field.dtypes)
    shape = (10,)
    new_shape = (2, 5)
    a = field.Random(shape, dtype=dtype)

    b = a.reshape(new_shape)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype

    b = np.reshape(a, new_shape)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_ravel(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 5)
    new_shape = (10,)
    a = field.Random(shape, dtype=dtype)
    b = np.ravel(a)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_flatten(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 5)
    new_shape = (10,)
    a = field.Random(shape, dtype=dtype)
    b = a.flatten()
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


###############################################################################
# Transpose-like operations
###############################################################################


def test_moveaxis(field):
    dtype = random.choice(field.dtypes)
    shape = (3, 4, 5)
    new_shape = (4, 3, 5)
    a = field.Random(shape, dtype=dtype)
    b = np.moveaxis(a, 0, 1)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_transpose(field):
    dtype = random.choice(field.dtypes)
    shape = (3, 4)
    new_shape = (4, 3)
    a = field.Random(shape, dtype=dtype)

    b = a.T
    assert b.shape == new_shape
    assert array_equal(b[0, :], a[:, 0])
    assert type(b) is field
    assert b.dtype == dtype

    b = np.transpose(a)
    assert b.shape == new_shape
    assert array_equal(b[0, :], a[:, 0])
    assert type(b) is field
    assert b.dtype == dtype


###############################################################################
# Changing number of dimensions
###############################################################################


def test_at_least1d(field):
    dtype = random.choice(field.dtypes)
    shape = ()
    new_shape = (1,)
    a = field.Random(shape, dtype=dtype)
    b = np.atleast_1d(a)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_at_least2d(field):
    dtype = random.choice(field.dtypes)
    shape = (10,)
    new_shape = (1, 10)
    a = field.Random(shape, dtype=dtype)
    b = np.atleast_2d(a)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_at_least3d(field):
    dtype = random.choice(field.dtypes)
    shape = (10,)
    new_shape = (1, 10, 1)
    a = field.Random(shape, dtype=dtype)
    b = np.atleast_3d(a)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_broadcast_to(field):
    dtype = random.choice(field.dtypes)
    shape = (3,)
    new_shape = (2, 3)
    a = field.Random(shape, dtype=dtype)
    b = np.broadcast_to(a, new_shape)
    assert b.shape == new_shape
    assert array_equal(b[1, :], a)
    assert type(b) is field
    assert b.dtype == dtype


def test_squeeze(field):
    dtype = random.choice(field.dtypes)
    shape = (1, 3, 1)
    new_shape = (3,)
    a = field.Random(shape, dtype=dtype)
    b = np.squeeze(a)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


###############################################################################
# Joining arrays
###############################################################################


def test_concatenate(field):
    dtype = random.choice(field.dtypes)
    shape1 = (2, 3)
    shape2 = (1, 3)
    new_shape = (3, 3)
    a1 = field.Random(shape1, dtype=dtype)
    a2 = field.Random(shape2, dtype=dtype)
    b = np.concatenate((a1, a2), axis=0)
    assert b.shape == new_shape
    assert array_equal(b[0:2, :], a1)
    assert array_equal(b[2:, :], a2)
    assert type(b) is field
    assert b.dtype == dtype


def test_vstack(field):
    dtype = random.choice(field.dtypes)
    shape1 = (3,)
    shape2 = (3,)
    new_shape = (2, 3)
    a1 = field.Random(shape1, dtype=dtype)
    a2 = field.Random(shape2, dtype=dtype)
    b = np.vstack((a1, a2))
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_hstack(field):
    dtype = random.choice(field.dtypes)
    shape1 = (3,)
    shape2 = (3,)
    new_shape = (6,)
    a1 = field.Random(shape1, dtype=dtype)
    a2 = field.Random(shape2, dtype=dtype)
    b = np.hstack((a1, a2))
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


###############################################################################
# Splitting arrays
###############################################################################


def test_split(field):
    dtype = random.choice(field.dtypes)
    shape = (6,)
    new_shape = (3,)
    a = field.Random(shape, dtype=dtype)
    b1, b2 = np.split(a, 2)
    assert b1.shape == new_shape
    assert type(b1) is field
    assert b1.dtype == dtype
    assert b2.shape == new_shape
    assert type(b2) is field
    assert b2.dtype == dtype


###############################################################################
# Tiling arrays
###############################################################################


def test_tile(field):
    dtype = random.choice(field.dtypes)
    shape = (3,)
    new_shape = (2, 6)
    a = field.Random(shape, dtype=dtype)
    b = np.tile(a, (2, 2))
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


def test_repeat(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 2)
    new_shape = (2, 6)
    a = field.Random(shape, dtype=dtype)
    b = np.repeat(a, 3, axis=1)
    assert b.shape == new_shape
    assert type(b) is field
    assert b.dtype == dtype


###############################################################################
# Adding and removing elements
###############################################################################


def test_delete(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 4)
    new_shape = (2, 3)
    a = field.Random(shape, dtype=dtype)
    b = np.delete(a, 1, axis=1)
    assert b.shape == new_shape
    assert array_equal(b[:, 0], a[:, 0])
    assert array_equal(b[:, 1:], a[:, 2:])
    assert type(b) is field
    assert b.dtype == dtype


def test_insert_field_element(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 4)
    new_shape = (2, 5)
    a = field.Random(shape, dtype=dtype)
    b = field.Random()
    c = np.insert(a, 1, b, axis=1)
    assert c.shape == new_shape
    assert array_equal(c[:, 0], a[:, 0])
    assert np.all(c[:, 1] == b)
    assert array_equal(c[:, 2:], a[:, 1:])
    assert type(c) is field
    assert c.dtype == dtype


def test_insert_int(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 4)
    new_shape = (2, 5)
    a = field.Random(shape, dtype=dtype)
    b = random.randint(0, field.order - 1)
    c = np.insert(a, 1, b, axis=1)
    assert c.shape == new_shape
    assert array_equal(c[:, 0], a[:, 0])
    assert np.all(c[:, 1] == b)
    assert array_equal(c[:, 2:], a[:, 1:])
    assert type(c) is field
    assert c.dtype == dtype


def test_insert_int_out_of_range(field):
    for dtype in [field.dtypes[0], field.dtypes[-1]]:
        shape = (2, 4)
        a = field.Random(shape, dtype=dtype)
        b = field.order
        with pytest.raises(ValueError):
            np.insert(a, 1, b, axis=1)


def test_insert_int_list(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 4)
    new_shape = (2, 5)
    a = field.Random(shape, dtype=dtype)
    b = [random.randint(0, field.order - 1) for _ in range(2)]
    c = np.insert(a, 1, b, axis=1)
    assert c.shape == new_shape
    assert array_equal(c[:, 0], a[:, 0])
    assert array_equal(c[:, 1], b)
    assert array_equal(c[:, 2:], a[:, 1:])
    assert type(c) is field
    assert c.dtype == dtype


def test_insert_int_array(field):
    dtype = field.dtypes[0]
    shape = (2, 4)
    new_shape = (2, 5)
    a = field.Random(shape, dtype=dtype)
    b = randint(0, field.order, 2, field.dtypes[-1])
    c = np.insert(a, 1, b, axis=1)
    assert c.shape == new_shape
    assert array_equal(c[:, 0], a[:, 0])
    assert array_equal(c[:, 1], b)
    assert array_equal(c[:, 2:], a[:, 1:])
    assert type(c) is field
    assert c.dtype == dtype


def test_insert_int_array_out_of_range(field):
    dtype = field.dtypes[0]
    shape = (2, 4)
    a = field.Random(shape, dtype=dtype)
    b = randint(field.order, field.order + 2, 2, field.dtypes[-1])
    with pytest.raises(ValueError):
        np.insert(a, 1, b, axis=1)


def test_append(field):
    dtype = random.choice(field.dtypes)
    shape1 = (2, 3)
    shape2 = (1, 3)
    new_shape = (3, 3)
    a1 = field.Random(shape1, dtype=dtype)
    a2 = field.Random(shape2, dtype=dtype)
    b = np.append(a1, a2, axis=0)
    assert b.shape == new_shape
    assert array_equal(b[0:2, :], a1)
    assert array_equal(b[2:, :], a2)
    assert type(b) is field
    assert b.dtype == dtype


def test_resize(field):
    dtype = random.choice(field.dtypes)
    shape = (3,)
    new_shape = (2, 3)
    a = field.Random(shape, dtype=dtype)

    b = np.resize(a, new_shape)
    assert b.shape == new_shape
    assert array_equal(b[0, :], a)
    assert array_equal(b[1, :], a)
    assert type(b) is field
    assert b.dtype == dtype

    # TODO: Why does c not "own its data"?
    # c = np.copy(a)
    # c.resize(new_shape)
    # assert c.shape == new_shape
    # assert array_equal(c[0,:], a)
    # assert array_equal(c[1,:], 0)  # NOTE: This is different than np.resize()
    # assert type(c) is field
    # assert c.dtype == dtype


def test_trim_zeros(field):
    dtype = random.choice(field.dtypes)
    shape = (5,)
    new_shape = (2,)
    a = field.Random(shape, low=1, dtype=dtype)
    a[0:2] = 0
    a[-1] = 0
    b = np.trim_zeros(a, trim="fb")
    assert b.shape == new_shape
    assert array_equal(b, a[2:-1])
    assert type(b) is field


def test_unique(field):
    dtype = random.choice(field.dtypes)
    size = field.order if field.order < 10 else 10
    a = field.Range(0, size, dtype=dtype)
    a[0] = 1  # Remove 0 element
    b = np.unique(a)
    assert array_equal(b, a[1:])
    assert type(b) is field


###############################################################################
# Rearranging elements
###############################################################################


def test_flip(field):
    dtype = random.choice(field.dtypes)
    shape = (3,)
    a = field.Random(shape, dtype=dtype)
    b = np.flip(a)
    assert array_equal(b, a[::-1])
    assert type(b) is field


def test_fliplr(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 3)
    a = field.Random(shape, dtype=dtype)
    b = np.fliplr(a)
    assert array_equal(b, a[:, ::-1])
    assert type(b) is field


def test_flipud(field):
    dtype = random.choice(field.dtypes)
    shape = (2, 3)
    a = field.Random(shape, dtype=dtype)
    b = np.flipud(a)
    assert array_equal(b, a[::-1, :])
    assert type(b) is field


def test_roll(field):
    dtype = random.choice(field.dtypes)
    shape = (10,)
    a = field.Random(shape, dtype=dtype)
    b = np.roll(a, 2)
    assert array_equal(b[0:2], a[-2:])
    assert array_equal(b[2:], a[0:-2])
    assert type(b) is field
