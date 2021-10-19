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
    "name", "characteristic", "degree", "order",
    "irreducible_poly", "is_primitive_poly", "primitive_element", "primitive_elements",
    "is_prime_field", "is_extension_field", "prime_subfield",
    "dtypes", "display_mode", "properties",
    "ufunc_mode", "ufunc_modes",
]

@pytest.mark.parametrize("attribute", ATTRIBUTES)
def test_cant_set_attribute(attribute):
    GF = galois.GF2
    with pytest.raises(AttributeError):
        setattr(GF, attribute, None)


def test_properties():
    GF = galois.GF(7)
    assert GF.properties == "GF(7):\n  characteristic: 7\n  degree: 1\n  order: 7\n  irreducible_poly: x + 4\n  is_primitive_poly: True\n  primitive_element: 3"

    GF = galois.GF(2**8)
    assert GF.properties == "GF(2^8):\n  characteristic: 2\n  degree: 8\n  order: 256\n  irreducible_poly: x^8 + x^4 + x^3 + x^2 + 1\n  is_primitive_poly: True\n  primitive_element: x"


def test_is_primitive_poly():
    """
    Verify the `is_primitive_poly` boolean is calculated correctly for fields constructed with explicitly-specified
    irreducible polynomials.
    """
    # GF(2^m) with integer dtype
    poly = galois.conway_poly(2, 32)
    GF = galois.GF(2**32, irreducible_poly=poly, primitive_element="x", verify=False)
    assert GF.is_primitive_poly == True

    # GF(2^m) with object dtype
    poly = galois.conway_poly(2, 100)
    GF = galois.GF(2**100, irreducible_poly=poly, primitive_element="x", verify=False)
    assert GF.is_primitive_poly == True

    # GF(p^m) with integer dtype
    poly = galois.conway_poly(3, 20)
    GF = galois.GF(3**20, irreducible_poly=poly, primitive_element="x", verify=False)
    assert GF.is_primitive_poly == True

    # GF(p^m) with object dtype
    poly = galois.conway_poly(3, 101)
    GF = galois.GF(3**101, irreducible_poly=poly, primitive_element="x", verify=False)
    assert GF.is_primitive_poly == True
