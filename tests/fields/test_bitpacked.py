import itertools
import operator as ops

import numpy as np
import pytest

import galois
from galois import GF2, GF2BP


def test_shape():
    a = GF2.Random((100, 100))
    a_ = np.packbits(a, axis=1)
    assert a_.shape == (100, 100)

    a_ = np.packbits(a, axis=0)
    assert a_.shape == (100, 100)


def test_repr():
    a = GF2(0)
    a_ = np.packbits(a)

    assert repr(a_) == "GF([0], order=2, bitpacked)"


def test_packbits_on_existing():
    a = GF2(0)
    a_ = np.packbits(a)
    assert a is not a_

    a__ = np.packbits(a_)
    assert a_ is a__


def test_unpackbits_on_different_axis():
    a = np.packbits(GF2([[1, 0], [0, 1]]))

    with pytest.raises(ValueError, match="along a different axis"):
        np.unpackbits(a, axis=1)


def test_unpackbits_with_different_axis_count():
    a = np.packbits(GF2([[1, 0], [0, 1]]))

    with pytest.raises(ValueError, match="different axis element count"):
        np.unpackbits(a, count=10)


def test_new_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="GF2BP is a custom bit-packed GF2 class with limited functionality."):
        GF2BP.Random((100, 100))


def test_new_axis():
    a = GF2.Random((10, 20))
    a_ = np.packbits(a)
    a__ = a_[:, None]
    assert a__.shape == a[:, None].shape
    assert np.array_equal(a__, a[:, None])


def test_galois_array_indexing():
    # Define a Galois field array
    GF = galois.GF(2)
    # Use an array length > GF2BP.BIT_WIDTH as a stronger test for indexing
    arr = GF([1, 0, 1, 1, 0, 0, 1, 1, 1])
    arr = np.packbits(arr)

    # 1. Basic Indexing
    assert arr[0] == GF(1)
    assert arr[2] == GF(1)

    # 2. Negative Indexing
    assert arr[-1] == GF(1)
    assert arr[-2] == GF(1)

    # 3. Slicing
    assert np.array_equal(arr[1:3], GF([0, 1]))
    assert np.array_equal(arr[:3], GF([1, 0, 1]))
    assert np.array_equal(arr[::2], GF([1, 1, 0, 1, 1]))
    assert np.array_equal(arr[::-1], GF([1, 1, 1, 0, 0, 1, 1, 0, 1]))
    assert np.array_equal(arr[-3:-1], GF([1, 1]))
    assert np.array_equal(arr[-1:-3:-1], GF([1, 1]))

    # 4. Multidimensional Indexing
    arr_2d = GF([[1, 0], [0, 1]])
    arr_2d = np.packbits(arr_2d)
    assert arr_2d[0, 1] == GF(0)
    assert np.array_equal(arr_2d[:, 1], GF([0, 1]))

    # 5. Boolean Indexing
    mask = np.array([True, False, True, False, False, False, True, True, True])
    assert np.array_equal(arr[mask], GF([1, 1, 1, 1, 1]))

    # 6. Fancy Indexing
    assert np.array_equal(arr[[0, 2, -1]], GF([1, 1, 1]))
    # first array is row indexing, second is column indexing
    assert np.array_equal(arr_2d[[0, 0], [1, 1]], GF([0, 0]))

    # 7. Ellipsis
    arr_3d = GF(np.random.randint(0, 2, (2, 3, 4)))
    arr_3d = np.packbits(arr_3d)
    shape_check = arr_3d[0, ..., 1].shape  # (3,)
    assert shape_check == (3,)

    # 8. Indexing with slice objects
    s = slice(1, 3)
    assert np.array_equal(arr[s], GF([0, 1]))

    # 9. Using np.newaxis
    reshaped = arr[:, np.newaxis]
    assert reshaped.shape == (9, 1)

    # 10. Indexing with np.ix_
    row_indices = np.array([0, 1])
    col_indices = np.array([0, 1])
    sub_matrix = arr_2d[np.ix_(row_indices, col_indices)]
    assert np.array_equal(sub_matrix, GF([[1, 0], [0, 1]]))

    # 11. Empty indexing
    arr = GF([1, 0, 1, 1])
    arr = np.packbits(arr)
    assert np.array_equal(arr[[]], GF([]))


def test_galois_array_setting():
    # Define a Galois field array
    GF = galois.GF(2)
    # Use an array length > GF2BP.BIT_WIDTH as a stronger test for indexing
    arr = GF([1, 0, 1, 1, 0, 0, 0, 0, 0])
    arr = np.packbits(arr)

    # 1. Basic Indexing
    arr[0] = GF(0)
    assert arr[0] == GF(0)
    arr[8] = GF(1)
    assert arr[8] == GF(1)

    # 2. Negative Indexing
    arr[-1] = GF(0)
    assert arr[-1] == GF(0)

    # 3. Slicing
    arr[1:3] = GF([1, 0])
    assert np.array_equal(arr, np.packbits(GF([0, 1, 0, 1, 0, 0, 0, 0, 0])))

    # 4. Multidimensional Indexing
    arr_2d = GF([[1, 0], [0, 1]])
    arr_2d = np.packbits(arr_2d)
    arr_2d[0, 1] = GF(1)
    assert arr_2d[0, 1] == GF(1)

    arr_2d[:, 1] = GF([0, 0])
    assert np.array_equal(arr_2d[:, 1], GF([0, 0]))

    # 5. Boolean Indexing
    arr = GF([1, 0, 1, 1])
    arr = np.packbits(arr)
    mask = np.array([True, False, True, False])
    arr[mask] = GF(0)
    assert np.array_equal(arr, np.packbits(GF([0, 0, 0, 1])))

    # 6. Fancy Indexing
    arr = GF([1, 0, 1, 1])
    arr = np.packbits(arr)
    indices = [0, 2, 3]
    arr[indices] = GF([0, 0, 0])
    assert np.array_equal(arr, np.packbits(GF([0, 0, 0, 0])))

    # 7. Ellipsis
    arr_3d = GF(np.random.randint(0, 2, (2, 3, 4)))
    arr_3d = np.packbits(arr_3d)
    arr_3d[0, ..., 1] = GF(1)
    assert np.array_equal(arr_3d[0, :, 1], GF([1, 1, 1]))

    # 8. Indexing with slice objects
    arr = GF([1, 0, 1, 1])
    arr = np.packbits(arr)
    s = slice(1, 3)
    arr[s] = GF([0, 0])
    assert np.array_equal(arr, np.packbits(GF([1, 0, 0, 1])))

    # 9. Using np.newaxis (reshaped array assignment)
    arr = GF([1, 0, 1, 1])
    arr = np.packbits(arr)

    reshaped = arr[:, np.newaxis]
    # Traditionally, this would be a view on arr's data, but for bitpacked arrays it will be a copy in which we make it
    # read-only. Let's verify that.
    assert not reshaped.flags.writeable

    reshaped = np.packbits(reshaped)
    reshaped[:, 0] = GF([0, 0, 0, 0])
    assert not np.array_equal(arr, reshaped)  # Verify that these are not aliasing the same data.
    assert np.array_equal(reshaped, np.packbits(GF([[0], [0], [0], [0]])))

    # 10. Indexing with np.ix_
    arr_2d = GF([[1, 0], [0, 1]])
    arr_2d = np.packbits(arr_2d)
    row_indices = np.array([0, 1])
    col_indices = np.array([0, 1])
    arr_2d[np.ix_(row_indices, col_indices)] = GF([[0, 0], [0, 0]])
    assert np.array_equal(arr_2d, np.packbits(GF([[0, 0], [0, 0]])))

    # 11. Empty indexing
    arr = GF([1, 0, 1, 1])
    arr = np.packbits(arr)
    arr[[]] = 0
    assert np.array_equal(arr, np.packbits(GF([1, 0, 1, 1])))


def test_negative():
    N = 10
    u = GF2.Random((N, N), seed=2)
    p = np.packbits(u)

    assert np.array_equal(np.negative(u), np.unpackbits(np.negative(p)))


def test_power():
    N = 10
    u = GF2.Random((N, N), seed=2)
    p = np.packbits(u)

    powers = np.random.randint(0, 2 * GF2.order, N)

    assert np.array_equal(np.power(u, powers), np.unpackbits(np.power(p, powers)))


def test_log():
    N = 10
    u = GF2.Ones((N, N))
    p = np.packbits(u)

    assert np.array_equal(np.log(u), np.unpackbits(np.log(p)))


def test_inv():
    N = 10
    u = GF2.Random((N, N), seed=2)
    p = np.packbits(u)
    assert np.array_equal(np.linalg.inv(u), np.unpackbits(np.linalg.inv(p)))


def test_arithmetic():
    size = (20, 10)
    a = np.random.randint(2, size=size, dtype=np.uint8)
    b = np.random.randint(2, size=size, dtype=np.uint8)
    vec = np.random.randint(2, size=size[1], dtype=np.uint8)

    a_gf2 = GF2(a)
    b_gf2 = GF2(b)
    c_gf2 = GF2(b.T)
    vec_gf2 = GF2(vec)

    for axis_a, axis_b in itertools.product((0, 1), repeat=2):
        a_gf2bp = np.packbits(a_gf2, axis=axis_a)
        b_gf2bp = np.packbits(b_gf2, axis=axis_b)
        c_gf2bp = np.packbits(b_gf2.T, axis=axis_b)
        vec_gf2bp = np.packbits(vec_gf2, axis=0)  # Only one axis for a vector

        # Addition
        assert np.array_equal(np.unpackbits(a_gf2bp + b_gf2bp), a_gf2 + b_gf2)

        # Multiplication
        assert np.array_equal(np.unpackbits(a_gf2bp * b_gf2bp), a_gf2 * b_gf2)

        # Matrix-vector product
        assert np.array_equal(np.unpackbits(a_gf2bp @ vec_gf2bp), a_gf2 @ vec_gf2)

        # Matrix-matrix product
        assert np.array_equal(np.unpackbits(a_gf2bp @ c_gf2bp), a_gf2 @ c_gf2)


def test_broadcasting():
    a = GF2(np.random.randint(0, 2, 10))
    b = GF2(np.random.randint(0, 2, 10))
    x = np.packbits(a)
    y = np.packbits(b)

    for op in [ops.add, ops.sub, ops.mul]:
        c = op(a, b)
        z = op(x, y)
        assert c.shape == z.shape == np.unpackbits(z).shape  # (10,)

    c = np.multiply.outer(a, b)
    z = np.multiply.outer(x, y)
    assert np.array_equal(np.unpackbits(z), c)
    assert c.shape == z.shape == np.unpackbits(z).shape  # (10, 10)


def test_advanced_broadcasting():
    a = GF2(np.random.randint(0, 2, (1, 2, 3)))
    b = GF2(np.random.randint(0, 2, (2, 2, 1)))
    x = np.packbits(a)
    y = np.packbits(b)

    for op in [ops.add, ops.sub, ops.mul]:
        c = op(a, b)
        z = op(x, y)
        assert np.array_equal(np.unpackbits(z), c)
        assert c.shape == z.shape == np.unpackbits(z).shape  # (2, 2, 3)

    c = np.multiply.outer(a, b)
    z = np.multiply.outer(x, y)
    assert np.array_equal(np.unpackbits(z), c)
    assert c.shape == z.shape == np.unpackbits(z).shape  # (1, 2, 3, 2, 2, 1)
