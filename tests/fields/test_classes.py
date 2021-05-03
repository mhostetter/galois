"""
A pytest module to test the class attributes of Galois field array classes.
"""
import pytest
import numpy as np

import galois


def test_ufunc_attributes(field_classes):
    GF, mode = field_classes["GF"], field_classes["mode"]
    if mode == "auto":
        if GF.order == 2:
            mode = "jit-calculate"
        elif GF.order <= 2**16:
            mode = "jit-lookup"
        else:
            mode = "jit-calculate"

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
    elif field.order == 7**3:
        assert field.dtypes == [np.uint16, np.uint32, np.int16, np.int32, np.int64]
    elif field.order == 109987**4:
        assert field.dtypes == [np.object_]
    else:
        raise AssertionError(f"There is an untested field, {field}")


ATTRIBUTES = [
    "characteristic", "default_ufunc_mode", "degree", "display_mode",
    "dtypes", "prime_subfield", "irreducible_poly", "is_extension_field", "is_prime_field",
    "is_primitive_poly", "name", "order", "primitive_element", "primitive_elements",
    "properties", "ufunc_mode", "ufunc_modes", "ufunc_target", "ufunc_targets",
]

@pytest.mark.parametrize("attribute", ATTRIBUTES)
def test_cant_set_attribute(attribute):
    GF = galois.GF2
    with pytest.raises(AttributeError):
        setattr(GF, attribute, None)
