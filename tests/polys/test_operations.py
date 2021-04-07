"""
A pytest module to test various Galois field polynomial operations.
"""
import random

import numpy as np
from numpy.core.defchararray import multiply
import pytest

import galois


def test_string():
    poly = galois.Poly([1,0,1,1])
    assert repr(poly) == str(poly) == "Poly(x^3 + x + 1, GF(2))"

    GF = galois.GF(7)
    poly = galois.Poly([5,0,3,1], field=GF)
    assert repr(poly) == str(poly) == "Poly(5x^3 + 3x + 1, GF(7))"

    GF = galois.GF(2**3)
    poly = galois.Poly([2,0,3,1], field=GF)
    assert repr(poly) == str(poly) == "Poly(2x^3 + 3x + 1, GF(2^3))"


def test_roots():
    GF = galois.GF(31)
    roots = GF.Random(5).tolist()
    poly = galois.Poly.Roots(roots, field=GF)
    assert set(poly.roots().tolist()) == set(roots)


def test_multiple_roots():
    GF = galois.GF(31)
    roots = [0, 3, 17, 24, 30]
    multiplicities = [7, 5, 9, 2, 11]
    all_roots = []
    for r, m in zip(roots, multiplicities):
        all_roots += [r,]*m

    poly = galois.Poly.Roots(all_roots, field=GF)
    r, m = poly.roots(multiplicity=True)
    assert np.array_equal(r, roots)
    assert np.array_equal(m, multiplicities)
    assert type(r) is GF
    assert type(m) is np.ndarray


def test_derivative():
    GF = galois.GF(3)
    p = galois.Poly.Degrees([6,0], field=GF)
    dp = p.derivative()
    dp_truth = galois.Poly(0, field=GF)
    assert dp == dp_truth
    assert type(dp) is galois.Poly
    assert dp.field is GF

    GF = galois.GF(7)
    p = galois.Poly([3,5,0,2], field=GF)
    dp = p.derivative()
    dp_truth = galois.Poly(GF([3,5,0]) * np.array([3,2,1]), field=GF)
    assert dp == dp_truth
    assert type(dp) is galois.Poly
    assert dp.field is GF


def test_multiple_derivatives():
    GF = galois.GF(31)
    p = galois.Poly.Random(5, field=GF)
    dp = p.derivative(3)
    assert dp == p.derivative().derivative().derivative()
    assert type(dp) is galois.Poly
    assert dp.field is GF


def test_integer():
    poly = galois.Poly([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,0,0,1])
    assert poly.integer == 4295000729

    poly = galois.Poly.Degrees([32,15,9,7,4,3,0])
    assert poly.integer == 4295000729


def test_order_default(field):
    c = [1, 0, 1, 1]
    p = galois.Poly(c)
    assert np.array_equal(p.coeffs, c)
    assert np.array_equal(p.coeffs_desc, c)
    assert np.array_equal(p.coeffs_asc, c[::-1])


def test_order_asc(field):
    c = [1, 0, 1, 1]
    p = galois.Poly(c, order="asc")
    assert np.array_equal(p.coeffs, c[::-1])
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


def test_poly_gcd():
    GF = galois.GF(7)
    a = galois.Poly.Roots([2,2,2,3,5], field=GF)
    b = galois.Poly.Roots([1,2,6], field=GF)
    gcd, x, y = galois.poly_gcd(a, b)
    assert a*x + b*y == gcd
