"""
A pytest module to test the class attributes of Galois field array classes.
"""
import numpy as np
import pytest

import galois


def test_ufunc_attributes(field_classes):
    GF, mode = field_classes["GF"], field_classes["mode"]
    if mode == "auto":
        if GF.order == 2:
            mode = "calculate"
        elif GF.order <= 2**16:
            mode = "lookup"
        else:
            mode = "calculate"

    assert GF.ufunc_target == "cpu"
    assert GF.ufunc_mode == mode


def test_dtypes(field):
    if field.order == 2:
        assert field.dtypes == [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]
    elif field.order == 2**2:
        assert field.dtypes == [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]
    elif field.order == 2**3:
        assert field.dtypes == [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]
    elif field.order == 2**8:
        assert field.dtypes == [np.uint8, np.uint16, np.uint32, np.int16, np.int32, np.int64]
    elif field.order == 2**32:
        assert field.dtypes == [np.uint32, np.int64]
    elif field.order == 2**100:
        assert field.dtypes == [np.object_]
    elif field.order == 5:
        assert field.dtypes == [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]
    elif field.order == 7:
        assert field.dtypes == [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]
    elif field.order == 31:
        assert field.dtypes == [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]
    elif field.order == 3191:
        assert field.dtypes == [np.uint16, np.uint32, np.int16, np.int32, np.int64]
    elif field.order == 2147483647:
        assert field.dtypes == [np.uint32, np.int32, np.int64]
    elif field.order == 36893488147419103183:
        assert field.dtypes == [np.object_]
    else:
        raise AssertionError(f"There is an untested field, {field}")


def test_cant_set_characteristic():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.characteristic = None


def test_cant_set_degree():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.degree = None


def test_cant_set_order():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.order = None


def test_cant_set_irreducible_poly():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.irreducible_poly = None
    assert GF.irreducible_poly is not GF._irreducible_poly


def test_cant_set_primitive_element():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.primitive_element = None
    assert GF.primitive_element is not GF._primitive_element


def test_cant_set_dtypes():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.dtypes = None
    assert GF.dtypes is not GF._dtypes


def test_cant_set_ufunc_mode():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.ufunc_mode = None


def test_cant_set_ufunc_target():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.ufunc_target = None


def test_cant_set_display_mode():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.display_mode = None


def test_cant_set_display_poly_var():
    GF = galois.GF2
    with pytest.raises(AttributeError):
        GF.display_poly_var = None
