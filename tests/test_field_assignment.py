"""
A pytest module to test Galois field array assignment.
"""
import numpy as np
import pytest

import galois


class TestConstantIndex:
    def test_valid(self, field):
        a = field.Random(10)
        a[0] = 1

    def test_invalid_type(self, field):
        a = field.Random(10)
        with pytest.raises(TypeError):
            a[0] = 1.0

    def test_out_of_range(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            a[0] = field.order


class TestSliceIndex:
    def test_constant_valid(self, field):
        a = field.Random(10)
        a[0:2] = 1

    def test_constant_invalid_type(self, field):
        a = field.Random(10)
        with pytest.raises(TypeError):
            a[0:2] = 1.0

    def test_constant_out_of_range(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            a[0:2] = field.order

    def test_list_valid(self, field):
        a = field.Random(10)
        a[0:2] = [1, 1]

    def test_list_invalid_type(self, field):
        a = field.Random(10)
        with pytest.raises(TypeError):
            a[0:2] = [1.0, 1]

    def test_list_out_of_range(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            a[0:2] = [field.order, 1]

    def test_array_valid(self, field):
        a = field.Random(10)
        a[0:2] = np.array([1, 1])

    def test_array_valid_small_dtype(self, field):
        a = field.Random(10)
        a[0:2] = np.array([1, 1], dtype=np.int8)

    def test_array_invalid_type(self, field):
        a = field.Random(10)
        with pytest.raises(TypeError):
            a[0:2] = np.array([1.0, 1])

    def test_array_out_of_range(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            a[0:2] = np.array([field.order, 1])


class Test2DSliceIndex:
    def test_list_valid(self, field):
        a = field.Random((10,10))
        a[0:2, 0:2] = [[1, 1], [1, 1]]

    def test_list_invalid_type(self, field):
        a = field.Random((10,10))
        with pytest.raises(TypeError):
            a[0:2, 0:2] = [[1.0, 1], [1, 1]]

    def test_list_out_of_range(self, field):
        a = field.Random((10,10))
        with pytest.raises(ValueError):
            a[0:2, 0:2] = [[field.order, 1], [1, 1]]

    def test_array_valid(self, field):
        a = field.Random((10,10))
        a[0:2, 0:2] = np.array([[1, 1], [1, 1]])

    def test_array_valid_small_dtype(self, field):
        a = field.Random((10,10))
        a[0:2, 0:2] = np.array([[1, 1], [1, 1]], dtype=np.int8)

    def test_array_invalid_type(self, field):
        a = field.Random((10,10))
        with pytest.raises(TypeError):
            a[0:2, 0:2] = np.array([[1.0, 1], [1, 1]])

    def test_array_out_of_range(self, field):
        a = field.Random((10,10))
        with pytest.raises(ValueError):
            a[0:2, 0:2] = np.array([[field.order, 1], [1, 1]])
