"""
A pytest module to test various Galois field polynomial operations.
"""
import random

import pytest
import numpy as np

import galois


def test_copy():
    p1 = galois.Poly([2,0,1,2], field=galois.GF(3))
    assert isinstance(p1, galois.poly.DensePoly)
    p2 = p1.copy()
    p2._coeffs[0] = 0
    assert np.array_equal(p1.coeffs, [2,0,1,2])

    p1 = galois.Poly([1,0,1,1])
    assert isinstance(p1, galois.poly.BinaryPoly)
    p2 = p1.copy()
    p2._integer = 27
    assert np.array_equal(p1.coeffs, [1,0,1,1])

    p1 = galois.Poly.Degrees([3000,1,0], [1,1,1])
    assert isinstance(p1, galois.poly.SparsePoly)
    p2 = p1.copy()
    p2._degrees[0] = 4000
    assert np.array_equal(p1.nonzero_degrees, [3000,1,0])


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


def test_roots_with_multiplicity():
    GF = galois.GF(31)
    roots = [0, 3, 17, 24, 30]
    multiplicities = [7, 5, 9, 2, 11]

    poly = galois.Poly.Roots(roots, multiplicities=multiplicities, field=GF)
    r, m = poly.roots(multiplicity=True)
    assert np.array_equal(r, roots)
    assert np.array_equal(m, multiplicities)
    assert type(r) is GF
    assert type(m) is np.ndarray


def test_derivative():
    GF = galois.GF(3)
    p = galois.Poly.Degrees([6,0], field=GF)
    dp = p.derivative()
    dp_truth = galois.Poly([0], field=GF)
    assert dp == dp_truth
    assert isinstance(dp, galois.Poly)
    assert dp.field is GF

    GF = galois.GF(7)
    p = galois.Poly([3,5,0,2], field=GF)
    dp = p.derivative()
    dp_truth = galois.Poly(GF([3,5,0]) * np.array([3,2,1]), field=GF)
    assert dp == dp_truth
    assert isinstance(dp, galois.Poly)
    assert dp.field is GF


def test_multiple_derivatives():
    GF = galois.GF(31)
    p = galois.Poly.Random(5, field=GF)
    dp = p.derivative(3)
    assert dp == p.derivative().derivative().derivative()
    assert isinstance(dp, galois.Poly)
    assert dp.field is GF


def test_integer():
    poly = galois.Poly([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,0,0,1])
    assert poly.integer == 4295000729

    poly = galois.Poly.Degrees([32,15,9,7,4,3,0])
    assert poly.integer == 4295000729


def test_equal(field):
    c = field.Random(6)
    c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    p1 = galois.Poly(c)
    p2 = galois.Poly(c.tolist(), field=field)
    assert p1 == p2
