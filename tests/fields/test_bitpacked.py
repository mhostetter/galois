import numpy as np
import galois

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

    print("All tests passed.")


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

    print("All set-indexing tests passed.")


if __name__ == "__main__":
    test_galois_array_indexing()
    test_galois_array_setting()
