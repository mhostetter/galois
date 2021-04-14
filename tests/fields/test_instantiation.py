"""
A pytest module to test instantiation of new Galois field arrays.
"""
import random

import pytest
import numpy as np

import galois

from ..helper import array_equal


DTYPES = galois.dtypes.DTYPES + [np.object_]


def test_cant_instantiate_GF():
    v = [0, 1, 0, 1]
    with pytest.raises(NotImplementedError):
        a = galois.GFArray(v)


class Test0D:
    @pytest.mark.parametrize("type1", [int, list, tuple, np.array, galois.GFArray])
    def test_new(self, field, type1):
        v = int(field.Random())
        vt = convert_0d(v, type1, field)
        a = field(vt)
        assert type(a) is field
        assert a == v

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array, galois.GFArray])
    def test_valid_dtype(self, field, type1):
        v = int(field.Random())
        vt = convert_0d(v, type1, field)
        dtype = valid_dtype(field)
        a = field(vt, dtype=dtype)
        assert type(a) is field
        assert a.dtype == dtype
        assert a == v

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array, galois.GFArray])
    def test_invalid_dtype(self, field, type1):
        v = int(field.Random())
        vt = convert_0d(v, type1, field)
        dtype = invalid_dtype(field)
        with pytest.raises(TypeError):
            a = field(vt, dtype=dtype)

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array])
    def test_non_integer(self, field, type1):
        v = float(field.order)
        vt = convert_0d(v, type1, field)
        with pytest.raises((TypeError, ValueError)):
            a = field(vt)

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array])
    def test_out_of_range_low(self, field, type1):
        v = -1
        vt = convert_0d(v, type1, field)
        with pytest.raises(ValueError):
            a = field(vt)

    @pytest.mark.parametrize("type1", [int, list, tuple, np.array])
    def test_out_of_range_high(self, field, type1):
        v = field.order
        vt = convert_0d(v, type1, field)
        with pytest.raises(ValueError):
            a = field(vt)

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
        assert a.shape == (1,1,1)


class Test1D:
    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_new(self, field, type1):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        a = field(vt)
        assert type(a) is field
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_valid_dtype(self, field, type1):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        dtype = valid_dtype(field)
        a = field(vt, dtype=dtype)
        assert type(a) is field
        assert a.dtype == dtype
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_invalid_dtype(self, field, type1):
        v = [int(field.Random()), int(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        dtype = invalid_dtype(field)
        with pytest.raises(TypeError):
            a = field(vt, dtype=dtype)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_non_integer(self, field, type1):
        v = [int(field.Random()), float(field.Random()), int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        with pytest.raises((TypeError, ValueError)):
            a = field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_low(self, field, type1):
        v = [int(field.Random()), -1, int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        with pytest.raises(ValueError):
            a = field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_high(self, field, type1):
        v = [int(field.Random()), field.order, int(field.Random()), int(field.Random())]
        vt = convert_1d(v, type1, field)
        with pytest.raises(ValueError):
            a = field(vt)

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
        assert a.shape == (1,1,4)


class Test2D:
    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_new(self, field, type1):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        a = field(vt)
        assert type(a) is field
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_valid_dtype(self, field, type1):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        dtype = valid_dtype(field)
        a = field(vt, dtype=dtype)
        assert type(a) is field
        assert a.dtype == dtype
        assert array_equal(a, v)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_invalid_dtype(self, field, type1):
        v = [[int(field.Random()), int(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        dtype = invalid_dtype(field)
        with pytest.raises(TypeError):
            a = field(vt, dtype=dtype)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_non_integer(self, field, type1):
        v = [[int(field.Random()), float(field.Random())], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        with pytest.raises((TypeError, ValueError)):
            a = field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_low(self, field, type1):
        v = [[int(field.Random()), -1], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        with pytest.raises(ValueError):
            a = field(vt)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_out_of_range_high(self, field, type1):
        v = [[int(field.Random()), field.order], [int(field.Random()), int(field.Random())]]
        vt = convert_2d(v, type1, field)
        with pytest.raises(ValueError):
            a = field(vt)

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
        assert a.shape == (1,2,2)


def convert_0d(v, type1, field):
    if type1 is int:
        vt = v
    elif type1 in [list, tuple]:
        vt = type1([v])
    elif type1 is np.array and field.dtypes == [np.object_]:
        vt = np.array(v, dtype=np.object_)
    elif type1 is np.array:
        vt = np.array(v)
    elif type1 is galois.GFArray:
        vt = field(v)
    else:
        raise NotImplementedError
    return vt


def convert_1d(v, type1, field):
    if type1 is galois.GFArray:
        vt = field(v)
    elif type1 is np.array and field.dtypes == [np.object_]:
        vt = np.array(v, dtype=np.object_)
    elif type1 is np.array:
        vt = np.array(v)
    else:
        vt = type1(v)
    return vt


def convert_2d(v, type1, field):
    if type1 is galois.GFArray:
        vt = field(v)
    elif type1 is np.array and field.dtypes == [np.object_]:
        vt = np.array(v, dtype=np.object_)
    elif type1 is np.array:
        vt = np.array(v)
    elif type1 in [list, tuple]:
        vt = type1([type1([b for b in a]) for a in v])
    else:
        raise NotImplementedError
    return vt


def valid_dtype(field):
    return random.choice(field.dtypes)


def invalid_dtype(field):
    return random.choice([dtype for dtype in DTYPES if dtype not in field.dtypes])
