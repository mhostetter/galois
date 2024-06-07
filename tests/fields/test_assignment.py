"""
A pytest module to test FieldArray assignment.
"""

import numpy as np
import pytest

import galois


class TestScalarIndex:
    """
    Tests for assigning to a FieldArray using a scalar index.
    """

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

    def test_always_int_object(self):
        # Ensure when assigning FieldArray elements to an array they are converted to ints
        GF = galois.GF(2**100)
        a = GF.Random(10)
        assert np.all(is_int(a))
        a[0] = GF(10)
        assert np.all(is_int(a))


class TestSliceIndex:
    """
    Tests for assigning to a FieldArray using a slice index.
    """

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

    def test_always_int_object(self):
        # Ensure when assigning FieldArray elements to an array they are converted to ints
        GF = galois.GF(2**100)
        a = GF.Random(10)
        assert np.all(is_int(a))
        a[0:3] = [GF(10), GF(20), GF(30)]
        assert np.all(is_int(a))


class Test2DSliceIndex:
    """
    Tests for assigning to a 2D FieldArray using a slice index.
    """

    def test_list_valid(self, field):
        a = field.Random((10, 10))
        a[0:2, 0:2] = [[1, 1], [1, 1]]

    def test_list_invalid_type(self, field):
        a = field.Random((10, 10))
        with pytest.raises(TypeError):
            a[0:2, 0:2] = [[1.0, 1], [1, 1]]

    def test_list_out_of_range(self, field):
        a = field.Random((10, 10))
        with pytest.raises(ValueError):
            a[0:2, 0:2] = [[field.order, 1], [1, 1]]

    def test_array_valid(self, field):
        a = field.Random((10, 10))
        a[0:2, 0:2] = np.array([[1, 1], [1, 1]])

    def test_array_valid_small_dtype(self, field):
        a = field.Random((10, 10))
        a[0:2, 0:2] = np.array([[1, 1], [1, 1]], dtype=np.int8)

    def test_array_invalid_type(self, field):
        a = field.Random((10, 10))
        with pytest.raises(TypeError):
            a[0:2, 0:2] = np.array([[1.0, 1], [1, 1]])

    def test_array_out_of_range(self, field):
        a = field.Random((10, 10))
        with pytest.raises(ValueError):
            a[0:2, 0:2] = np.array([[field.order, 1], [1, 1]])

    def test_always_int_object(self):
        # Ensure when assigning FieldArray elements to an array they are converted to ints
        GF = galois.GF(2**100)
        a = GF.Random((10, 10))
        assert np.all(is_int(a))
        a[0:2, 0:2] = [[GF(10), GF(20)], [GF(30), GF(40)]]
        assert np.all(is_int(a))


is_int = np.vectorize(lambda element: isinstance(element, int))
