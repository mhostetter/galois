"""
A pytest module to test instantiation of new FieldArrays.
"""

import numpy as np
import pytest

import galois

from .conftest import array_equal, invalid_dtype, valid_dtype


def test_cant_instantiate_GF():
    v = [0, 1, 0, 1]
    with pytest.raises(NotImplementedError):
        galois.FieldArray(v)


def test_element_like_conversion():
    GF = galois.GF(3**5)
    assert np.array_equal(GF(17), GF("x^2 + 2x + 2"))
    assert np.array_equal(GF([[17, 4], [148, 205]]), GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]]))


class Test0D:
    """
    Tests for instantiating 0D arrays.
    """

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array, galois.FieldArray])
    def test_new(self, field, type1):
        v = int(field.Random())
        vt = convert_0d(v, type1, field)
        a = field(vt)
        assert type(a) is field
        assert a == v

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array, galois.FieldArray])
    def test_valid_dtype(self, field, type1):
        v = int(field.Random())
        vt = convert_0d(v, type1, field)
        dtype = valid_dtype(field)
        a = field(vt, dtype=dtype)
        assert type(a) is field
        assert a.dtype == dtype
        assert a == v

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array, galois.FieldArray])
    def test_invalid_dtype(self, field, type1):
        v = int(field.Random())
        vt = convert_0d(v, type1, field)
        dtype = invalid_dtype(field)
        with pytest.raises(TypeError):
            field(vt, dtype=dtype)

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array])
    def test_non_integer(self, field, type1):
        v = float(field.order)
        vt = convert_0d(v, type1, field)
        with pytest.raises((TypeError, ValueError)):
            field(vt)

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array])
    def test_out_of_range_low(self, field, type1):
        v = -1
        vt = convert_0d(v, type1, field)
        with pytest.raises(ValueError):
            field(vt)

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array])
    def test_out_of_range_high(self, field, type1):
        v = field.order
        vt = convert_0d(v, type1, field)
        with pytest.raises(ValueError):
            field(vt)

    def test_copy_true(self, field):
        v = int(field.Random(low=1))
        va = np.array(v, dtype=field.dtypes[0])
        a = field(va, copy=True)
        assert type(a) is field
        assert array_equal(a, v)
        va = 1  # Change original array
        assert array_equal(a, v)

    def test_default_order_c(self, field):
        v = int(field.Random())
        va = np.array(v, order="C", dtype=field.dtypes[0])
        a = field(va)  # Default order is "K" which keeps current
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_default_order_f(self, field):
        v = int(field.Random())
        va = np.array(v, order="F", dtype=field.dtypes[0])
        a = field(va)  # Default order is "K" which keeps current
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_order_c(self, field):
        v = int(field.Random())
        va = np.array(v, order="F", dtype=field.dtypes[0])
        a = field(va, order="C")
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_order_f(self, field):
        v = int(field.Random())
        va = np.array(v, order="C", dtype=field.dtypes[0])
        a = field(va, order="F")
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_ndmin(self, field):
        v = int(field.Random())
        a = field(v, ndmin=3)
        assert type(a) is field
        assert a.shape == (1, 1, 1)


class Test1D:
    """
    Tests for instantiating 1D arrays.
    """

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
    def test_new(self, field, type1):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        a = field(vt)
        assert type(a) is field
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
    def test_valid_dtype(self, field, type1):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        dtype = valid_dtype(field)
        a = field(vt, dtype=dtype)
        assert type(a) is field
        assert a.dtype == dtype
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
    def test_invalid_dtype(self, field, type1):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        dtype = invalid_dtype(field)
        with pytest.raises(TypeError):
            field(vt, dtype=dtype)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_non_integer(self, field, type1):
        v = [int(field.Random()), float(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        with pytest.raises((TypeError, ValueError)):
            field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_low(self, field, type1):
        v = [int(field.Random()), -1, int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        with pytest.raises(ValueError):
            field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_high(self, field, type1):
        v = [int(field.Random()), field.order, int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        with pytest.raises(ValueError):
            field(vt)

    def test_copy_true(self, field):
        v = [int(field.Random(low=1)), int(field.Random()), int(field.Random()), int(field.Random())]
        va = np.array(v, dtype=field.dtypes[0])
        a = field(va, copy=True)
        assert type(a) is field
        assert array_equal(a, v)
        va[0] = 0  # Change original array
        assert array_equal(a, v)

    def test_default_order_c(self, field):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        va = np.array(v, order="C", dtype=field.dtypes[0])
        a = field(va)  # Default order is "K" which keeps current
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_default_order_f(self, field):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        va = np.array(v, order="F", dtype=field.dtypes[0])
        a = field(va)  # Default order is "K" which keeps current
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_order_c(self, field):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        va = np.array(v, order="F", dtype=field.dtypes[0])
        a = field(va, order="C")
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_order_f(self, field):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        va = np.array(v, order="C", dtype=field.dtypes[0])
        a = field(va, order="F")
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_ndmin(self, field):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        a = field(v, ndmin=3)
        assert type(a) is field
        assert a.shape == (1, 1, 4)


class Test2D:
    """
    Tests for instantiating 2D arrays.
    """

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
    def test_new(self, field, type1):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        a = field(vt)
        assert type(a) is field
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
    def test_valid_dtype(self, field, type1):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        dtype = valid_dtype(field)
        a = field(vt, dtype=dtype)
        assert type(a) is field
        assert a.dtype == dtype
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
    def test_invalid_dtype(self, field, type1):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        dtype = invalid_dtype(field)
        with pytest.raises(TypeError):
            field(vt, dtype=dtype)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_non_integer(self, field, type1):
        v = [[int(field.Random()), float(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        with pytest.raises((TypeError, ValueError)):
            field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_low(self, field, type1):
        v = [[int(field.Random()), -1], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        with pytest.raises(ValueError):
            field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_high(self, field, type1):
        v = [[int(field.Random()), field.order], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        with pytest.raises(ValueError):
            field(vt)

    def test_copy_true(self, field):
        v = [[int(field.Random(low=1)), int(field.Random())], [int(field.Random()), int(field.Random())]]
        va = np.array(v, dtype=field.dtypes[0])
        a = field(va, copy=True)
        assert type(a) is field
        assert array_equal(a, v)
        va[0][0] = 0  # Change original array
        assert array_equal(a, v)

    def test_default_order_c(self, field):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        va = np.array(v, order="C", dtype=field.dtypes[0])
        a = field(va)  # Default order is "K" which keeps current
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert not a.flags["F_CONTIGUOUS"]

    def test_default_order_f(self, field):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        va = np.array(v, order="F", dtype=field.dtypes[0])
        a = field(va)  # Default order is "K" which keeps current
        assert type(a) is field
        assert not a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_order_c(self, field):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        va = np.array(v, order="F", dtype=field.dtypes[0])
        a = field(va, order="C")
        assert type(a) is field
        assert a.flags["C_CONTIGUOUS"]
        assert not a.flags["F_CONTIGUOUS"]

    def test_order_f(self, field):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        va = np.array(v, order="C", dtype=field.dtypes[0])
        a = field(va, order="F")
        assert type(a) is field
        assert not a.flags["C_CONTIGUOUS"]
        assert a.flags["F_CONTIGUOUS"]

    def test_ndmin(self, field):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        a = field(v, ndmin=3)
        assert type(a) is field
        assert a.shape == (1, 2, 2)


def convert_0d(v, type1, field):
    if type1 is int:
        vt = v
    elif type1 in [list, tuple]:
        vt = type1([v])
    elif type1 is np.array and field.dtypes == [np.object_]:
        vt = np.array(v, dtype=np.object_)
    elif type1 is np.array:
        vt = np.array(v)
    elif type1 is galois.FieldArray:
        vt = field(v)
    else:
        raise NotImplementedError
    return vt


def convert_1d(v, type1, field):
    if type1 is galois.FieldArray:
        vt = field(v)
    elif type1 is np.array and field.dtypes == [np.object_]:
        vt = np.array(v, dtype=np.object_)
    elif type1 is np.array:
        vt = np.array(v)
    else:
        vt = type1(v)
    return vt


def convert_2d(v, type1, field):
    if type1 is galois.FieldArray:
        vt = field(v)
    elif type1 is np.array and field.dtypes == [np.object_]:
        vt = np.array(v, dtype=np.object_)
    elif type1 is np.array:
        vt = np.array(v)
    elif type1 in [list, tuple]:
        vt = type1([type1(a) for a in v])
    else:
        raise NotImplementedError
    return vt
