"""
A pytest module to test instantiation of new Galois field arrays.
"""
import numpy as np
import pytest

import galois

from .helper import ALL_DTYPES


class TestAbstractClasses:
    def test_cant_instantiate_GF2m(self):
        v = [0, 1, 0, 1]
        with pytest.raises(NotImplementedError):
            a = galois.GF2m(v)

    def test_cant_instantiate_GFp(self):
        v = [0, 1, 0, 1]
        with pytest.raises(NotImplementedError):
            a = galois.GFp(v)


class TestList:
    def test_int(fself, field):
        v = [0, 1, 0, 1]
        a = field(v)
        assert type(a) is field

    def test_int_valid_dtypes(fself, field):
        v = [0, 1, 0, 1]
        for dtype in field.dtypes:
            a = field(v, dtype=dtype)
            assert a.dtype == dtype

    def test_int_invalid_dtypes(fself, field):
        v = [0, 1, 0, 1]
        for dtype in [d for d in ALL_DTYPES if d not in field.dtypes]:
            with pytest.raises(TypeError):
                a = field(v, dtype=dtype)

    def test_int_out_of_range(fself, field):
        v = [0, 1, 0, -1]
        with pytest.raises(ValueError):
            a = field(v)

        v = [0, 1, 0, field.order]
        with pytest.raises(ValueError):
            a = field(v)

    def test_float_valid_values(self, field):
        # Tests float values that could be coerced to field elements
        v = [0.0, 1.0, 0.0 , 1.0]
        with pytest.raises(TypeError):
            a = field(v)

    def test_float_invalid_values(self, field):
        # Tests float values that could not be coerced to field elements
        v = [0.1, field.order-0.8, 0.1, field.order-0.8]
        with pytest.raises(TypeError):
            a = field(v)


class TestArray:
    def test_int(self, field):
        v = np.array([0, 1, 0, 1])
        a = field(v)
        assert type(a) is field

    def test_valid_dtypes(self, field):
        v = np.array([0, 1, 0, 1])
        for dtype in field.dtypes:
            a = field(v, dtype=dtype)
            assert type(a) is field
            assert a.dtype == dtype

    def test_invalid_dtypes(self, field):
        v = np.array([0, 1, 0, 1])
        for dtype in [d for d in ALL_DTYPES if d not in field.dtypes]:
            with pytest.raises(TypeError):
                a = field(v, dtype=dtype)

    def test_int_out_of_range(self, field):
        v = np.array([0, 1, 0, field.order])
        with pytest.raises(ValueError):
            a = field(v)
        v = np.array([0,1,0,-1])
        with pytest.raises(ValueError):
            a = field(v)

    def test_float_valid_values(self, field):
        # Tests float values that could be coerced to field elements
        v = np.array([0.0, 1.0, 0.0 , 1.0])
        with pytest.raises(TypeError):
            a = field(v)

    def test_float_invalid_values(self, field):
        # Tests float values that could not be coerced to field elements
        v = np.array([0.1, field.order-0.8, 0.1, field.order-0.8])
        with pytest.raises(TypeError):
            a = field(v)


class TestConstructors:
    def test_zeros(self, field):
        a = field.Zeros(10)
        assert np.all(a == 0)
        assert type(a) is field
        assert a.dtype == field.dtypes[0]

    def test_zeros_valid_dtypes(self, field):
        for dtype in field.dtypes:
            a = field.Zeros(10, dtype=dtype)
            assert np.all(a == 0)
            assert type(a) is field
            assert a.dtype == dtype

    def test_zeros_invalid_dtypes(self, field):
        for dtype in [d for d in ALL_DTYPES if d not in field.dtypes]:
            with pytest.raises(TypeError):
                a = field.Zeros(10, dtype=dtype)

    def test_ones(self, field):
        a = field.Ones(10)
        assert np.all(a == 1)
        assert type(a) is field
        assert a.dtype == field.dtypes[0]

    def test_ones_valid_dtypes(self, field):
        for dtype in field.dtypes:
            a = field.Ones(10, dtype=dtype)
            assert np.all(a == 1)
            assert type(a) is field
            assert a.dtype == dtype

    def test_ones_invalid_dtypes(self, field):
        for dtype in [d for d in ALL_DTYPES if d not in field.dtypes]:
            with pytest.raises(TypeError):
                a = field.Ones(10, dtype=dtype)

    def test_random(self, field):
        a = field.Random(10)
        assert np.all(a >= 0) and np.all(a < field.order)
        assert type(a) is field
        assert a.dtype == field.dtypes[0]

    def test_random_valid_dtypes(self, field):
        for dtype in field.dtypes:
            a = field.Random(10, dtype=dtype)
            assert np.all(a >= 0) and np.all(a < field.order)
            assert type(a) is field
            assert a.dtype == dtype

    def test_random_invalid_dtypes(self, field):
        for dtype in [d for d in ALL_DTYPES if d not in field.dtypes]:
            with pytest.raises(TypeError):
                a = field.Random(10, dtype=dtype)

    def test_random_element(self, field):
        a = field.Random()
        assert np.all(a >= 0) and np.all(a < field.order)
        assert type(a) is field
        assert a.dtype == field.dtypes[0]

    def test_random_element_valid_dtypes(self, field):
        for dtype in field.dtypes:
            a = field.Random(dtype=dtype)
            assert np.all(a >= 0) and np.all(a < field.order)
            assert type(a) is field
            assert a.dtype == dtype

    def test_random_element_invalid_dtypes(self, field):
        for dtype in [d for d in ALL_DTYPES if d not in field.dtypes]:
            with pytest.raises(TypeError):
                a = field.Random(dtype=dtype)
