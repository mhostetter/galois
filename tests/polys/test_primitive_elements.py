"""
A pytest module to test generating primitive elements.
"""
import random

import pytest

import galois

from .luts.primitive_elements_2 import (
    PRIMITIVE_ELEMENTS_2_2,
    PRIMITIVE_ELEMENTS_2_3,
    PRIMITIVE_ELEMENTS_2_4,
    PRIMITIVE_ELEMENTS_2_5,
    PRIMITIVE_ELEMENTS_2_6,
)
from .luts.primitive_elements_3 import (
    PRIMITIVE_ELEMENTS_3_2,
    PRIMITIVE_ELEMENTS_3_3,
    PRIMITIVE_ELEMENTS_3_4,
)
from .luts.primitive_elements_5 import (
    PRIMITIVE_ELEMENTS_5_2,
    PRIMITIVE_ELEMENTS_5_3,
    PRIMITIVE_ELEMENTS_5_4,
)

PARAMS = [
    (2, 2, PRIMITIVE_ELEMENTS_2_2),
    (2, 3, PRIMITIVE_ELEMENTS_2_3),
    (2, 4, PRIMITIVE_ELEMENTS_2_4),
    (2, 5, PRIMITIVE_ELEMENTS_2_5),
    (2, 6, PRIMITIVE_ELEMENTS_2_6),
    (3, 2, PRIMITIVE_ELEMENTS_3_2),
    (3, 3, PRIMITIVE_ELEMENTS_3_3),
    (3, 4, PRIMITIVE_ELEMENTS_3_4),
    (5, 2, PRIMITIVE_ELEMENTS_5_2),
    (5, 3, PRIMITIVE_ELEMENTS_5_3),
    (5, 4, PRIMITIVE_ELEMENTS_5_4),
]


def test_primitive_element_exceptions():
    p = galois.conway_poly(2, 8)

    with pytest.raises(TypeError):
        galois.primitive_element(p.coeffs)
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Random(2) * galois.Poly.Random(2))
    with pytest.raises(ValueError):
        galois.primitive_element(p, method="invalid")
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Str("x^4"))


@pytest.mark.parametrize("characteristic,degree,elements", PARAMS)
def test_primitive_element_min(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p) == elements[0]


@pytest.mark.parametrize("characteristic,degree,elements", PARAMS)
def test_primitive_element_max(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p, method="max") == elements[-1]


@pytest.mark.parametrize("characteristic,degree,elements", PARAMS)
def test_primitive_element_random(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p, method="max") in elements


def test_primitive_elements_exceptions():
    p = galois.conway_poly(2, 8)

    with pytest.raises(TypeError):
        galois.primitive_elements(float(int(p)))
    with pytest.raises(ValueError):
        galois.primitive_elements(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.primitive_elements(galois.Poly.Random(2) * galois.Poly.Random(2))


@pytest.mark.parametrize("characteristic,degree,elements", PARAMS)
def test_primitive_elements(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_elements(p) == elements


def test_is_primitive_element_exceptions():
    e = galois.Poly([1, 0, 1, 1])
    f = galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1])

    with pytest.raises(TypeError):
        galois.is_primitive_element(float(int(e)), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]))
    with pytest.raises(TypeError):
        galois.is_primitive_element(e, f.coeffs)
    with pytest.raises(ValueError):
        galois.is_primitive_element(galois.Poly([1, 0, 1, 1]), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1], field=galois.GF(3)))
    with pytest.raises(ValueError):
        galois.is_primitive_element(galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]))


@pytest.mark.parametrize("characteristic,degree,elements", PARAMS)
def test_is_primitive_element(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert all(galois.is_primitive_element(e, p) for e in elements)


@pytest.mark.parametrize("characteristic,degree,elements", PARAMS)
def test_is_not_primitive_element(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    while True:
        e = galois.Poly.Int(random.randint(0, characteristic**degree - 1), field=galois.GF(characteristic))
        if e not in elements:
            break
    assert not galois.is_primitive_element(e, p)
