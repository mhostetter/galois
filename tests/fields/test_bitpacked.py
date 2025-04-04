import numpy as np
import galois
from galois import GF2
import operator as ops


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


def test_galois_array_indexing():
    # Define a Galois field array
    GF = galois.GF(2)
    arr = GF([1, 0, 1, 1])
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
    assert np.array_equal(arr[::2], GF([1, 1]))
    assert np.array_equal(arr[::-1], GF([1, 1, 0, 1]))

    # 4. Multidimensional Indexing
    arr_2d = GF([[1, 0], [0, 1]])
    arr_2d = np.packbits(arr_2d)
    assert arr_2d[0, 1] == GF(0)
    assert np.array_equal(arr_2d[:, 1], GF([0, 1]))

    # 5. Boolean Indexing
    mask = np.array([True, False, True, False])
    assert np.array_equal(arr[mask], GF([1, 1]))

    # 6. Fancy Indexing
    indices = [0, 2, 3]
    assert np.array_equal(arr[indices], GF([1, 1, 1]))

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
    assert reshaped.shape == (4, 1)

    # 10. Indexing with np.ix_
    row_indices = np.array([0, 1])
    col_indices = np.array([0, 1])
    sub_matrix = arr_2d[np.ix_(row_indices, col_indices)]
    assert np.array_equal(sub_matrix, GF([[1, 0], [0, 1]]))

    # TODO: Do we want to support this function?
    # 11. Indexing with np.take
    # taken = np.take(arr, [0, 2])
    # assert np.array_equal(taken, GF([1, 1]))

def test_galois_array_setting():
    # Define a Galois field array
    GF = galois.GF(2)
    arr = GF([1, 0, 1, 1])
    arr = np.packbits(arr)

    # 1. Basic Indexing
    arr[0] = GF(0)
    assert arr[0] == GF(0)

    # 2. Negative Indexing
    arr[-1] = GF(0)
    assert arr[-1] == GF(0)

    # 3. Slicing
    arr[1:3] = GF([1, 0])
    assert np.array_equal(arr, np.packbits(GF([0, 1, 0, 0])))

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
    reshaped = arr[:, np.newaxis]  # should this be using arr's data (as would be the case without packbits) or a new array?
    reshaped = np.packbits(reshaped)
    reshaped[:, 0] = GF([0, 0, 0, 0])
    # assert np.array_equal(arr, np.packbits(GF([0, 0, 0, 0])))
    assert np.array_equal(reshaped, np.packbits(GF([[0], [0], [0], [0]])))

    # 10. Indexing with np.ix_
    arr_2d = GF([[1, 0], [0, 1]])
    arr_2d = np.packbits(arr_2d)
    row_indices = np.array([0, 1])
    col_indices = np.array([0, 1])
    arr_2d[np.ix_(row_indices, col_indices)] = GF([[0, 0], [0, 0]])
    assert np.array_equal(arr_2d, np.packbits(GF([[0, 0], [0, 0]])))

def test_inv():
    N = 10
    u = GF2.Random((N, N), seed=2)
    p = np.packbits(u)
    # print(x.get_unpacked_slice(1))
    # index = np.index_exp[:,1:4:2]
    # index = np.index_exp[[0,1], [0, 1]]
    # print(a)
    # print(a[index])
    # print(x.get_unpacked_slice(index))
    print(np.linalg.inv(u))
    print(np.unpackbits(np.linalg.inv(p)))
    assert np.array_equal(np.linalg.inv(u), np.unpackbits(np.linalg.inv(p)))

def test_arithmetic():
    size = (20, 10)
    cm = np.random.randint(2, size=size, dtype=np.uint8)
    cm2 = np.random.randint(2, size=size, dtype=np.uint8)
    vec = np.random.randint(2, size=size[1], dtype=np.uint8)

    cm_GF2 = GF2(cm)
    cm2_GF2 = GF2(cm2)
    cm3_GF2 = GF2(cm2.T)
    vec_GF2 = GF2(vec)

    cm_gf2bp = np.packbits(cm_GF2)
    cm2_gf2bp = np.packbits(cm2_GF2)
    cm3_gf2bp = np.packbits(cm2_GF2.T)
    vec_gf2bp = np.packbits(vec_GF2)

    # Addition
    assert np.array_equal(np.unpackbits(cm_gf2bp + cm2_gf2bp), cm_GF2 + cm2_GF2)

    # Multiplication
    assert np.array_equal(np.unpackbits(cm_gf2bp * cm2_gf2bp), cm_GF2 * cm2_GF2)

    # Matrix-vector product
    assert np.array_equal(np.unpackbits(cm_gf2bp @ vec_gf2bp), cm_GF2 @ vec_GF2)

    # Matrix-matrix product
    assert np.array_equal(np.unpackbits(cm_gf2bp @ cm3_gf2bp), cm_GF2 @ cm3_GF2)

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
    print(c.shape)
    print(z.shape)
    assert np.array_equal(np.unpackbits(z), c)
    assert c.shape == z.shape == np.unpackbits(z).shape  # (1, 2, 3, 2, 2, 1)
