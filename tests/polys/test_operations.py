"""
A pytest module to test various Galois field polynomial operations.
"""
import random

import numpy as np
import pytest

import galois


def test_string():
    poly = galois.Poly([1,0,1,1])
    assert repr(poly) == str(poly) == "Poly(x^3 + x + 1, GF(2))"

    GF = galois.GF_factory(7, 1)
    poly = galois.Poly([5,0,3,1], field=GF)
    assert repr(poly) == str(poly) == "Poly(5x^3 + 3x + 1, GF(7))"

    GF = galois.GF_factory(2, 3)
    poly = galois.Poly([2,0,3,1], field=GF)
    assert repr(poly) == str(poly) == "Poly(2x^3 + 3x + 1, GF(2^3))"


def test_roots():
    GF = galois.GF_factory(31, 1)
    roots = GF.Random(5).tolist()
    poly = galois.Poly.Roots(roots, field=GF)
    assert set(poly.roots().tolist()) == set(roots)


def test_integer():
    poly = galois.Poly([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,0,0,1])
    assert poly.integer == 4295000729

    poly = galois.Poly.Degrees([32,15,9,7,4,3,0])
    assert poly.integer == 4295000729


def test_order_default(field):
    c = [1, 0, 1, 1]
    p = galois.Poly(c)
    assert p.order == "desc"
    assert np.array_equal(p.coeffs, c)
    assert np.array_equal(p.coeffs_desc, c)
    assert np.array_equal(p.coeffs_asc, c[::-1])


def test_order_asc(field):
    c = [1, 0, 1, 1]
    p = galois.Poly(c, order="asc")
    assert p.order == "asc"
    assert np.array_equal(p.coeffs, c)
    assert np.array_equal(p.coeffs_desc, c[::-1])
    assert np.array_equal(p.coeffs_asc, c)


def test_order_invalid(field):
    c = [1, 0, 1, 1]
    with pytest.raises(ValueError):
        p = galois.Poly(c, order="left-to-right")


def test_update_coeffs_field(field):
    c = field.Random(6)
    c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    p = galois.Poly(c)
    assert np.array_equal(p.coeffs, c)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert type(p.coeffs) is field

    c2 = field.Random(3)
    c2[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    p.coeffs = c2
    assert np.array_equal(p.coeffs, c2)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert type(p.coeffs) is field


def test_update_coeffs_list(field):
    c = field.Random(6)
    c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    l = c.tolist()
    p = galois.Poly(l, field=field)
    assert np.array_equal(p.coeffs, c)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert type(p.coeffs) is field

    c2 = [random.randint(0, field.order - 1) for _ in range(3)]
    c2[0] = random.randint(1, field.order - 1)  # Ensure leading coefficient is non-zero
    with pytest.raises(TypeError):
        p.coeffs = c2


def test_equal(field):
    c = field.Random(6)
    c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    p1 = galois.Poly(c)
    p2 = galois.Poly(c.tolist(), field=field)
    assert p1 == p2
