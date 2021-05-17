"""
A pytest module to test the class attributes of Galois field array classes.
"""
import pytest
import numpy as np

import galois


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
    "structure", "short_name", "name",
    "characteristic", "degree", "order",
    "irreducible_poly", "is_primitive_poly", "primitive_element", "primitive_elements",
    "is_prime_field", "is_extension_field", "prime_subfield",
    "dtypes", "display_mode", "properties",
    "ufunc_mode", "ufunc_modes", "ufunc_target", "ufunc_targets",
]

@pytest.mark.parametrize("attribute", ATTRIBUTES)
def test_cant_set_attribute(attribute):
    GF = galois.GF2
    with pytest.raises(AttributeError):
        setattr(GF, attribute, None)
