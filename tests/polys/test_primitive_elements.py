"""
A pytest module to test generating primitive elements.
"""

import random

import pytest

import galois

from .luts.primitive_elements import PRIMITIVE_ELEMENTS


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


@pytest.mark.parametrize("characteristic,degree,elements", PRIMITIVE_ELEMENTS)
def test_primitive_element_min(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p) == elements[0]


@pytest.mark.parametrize("characteristic,degree,elements", PRIMITIVE_ELEMENTS)
def test_primitive_element_max(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p, method="max") == elements[-1]


@pytest.mark.parametrize("characteristic,degree,elements", PRIMITIVE_ELEMENTS)
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


@pytest.mark.parametrize("characteristic,degree,elements", PRIMITIVE_ELEMENTS)
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
        galois.is_primitive_element(
            galois.Poly([1, 0, 1, 1]), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1], field=galois.GF(3))
        )
    with pytest.raises(ValueError):
        galois.is_primitive_element(galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]))


@pytest.mark.parametrize("characteristic,degree,elements", PRIMITIVE_ELEMENTS)
def test_is_primitive_element(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    assert all(galois.is_primitive_element(e, p) for e in elements)


@pytest.mark.parametrize("characteristic,degree,elements", PRIMITIVE_ELEMENTS)
def test_is_not_primitive_element(characteristic, degree, elements):
    p = galois.GF(characteristic**degree).irreducible_poly
    while True:
        e = galois.Poly.Int(random.randint(0, characteristic**degree - 1), field=galois.GF(characteristic))
        if e not in elements:
            break
    assert not galois.is_primitive_element(e, p)
