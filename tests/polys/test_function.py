"""
A pytest module to test functions on Galois field polynomials.
"""
import pytest
import numpy as np

import galois


def test_poly_gcd():
    a = galois.Poly.Random(5)
    b = galois.Poly.Zero()
    gcd, x, y = galois.poly_gcd(a, b)
    assert gcd == a
    assert a*x + b*y == gcd

    a = galois.Poly.Zero()
    b = galois.Poly.Random(5)
    gcd, x, y = galois.poly_gcd(a, b)
    assert gcd == b
    assert a*x + b*y == gcd

    GF = galois.GF(7)
    a = galois.Poly.Roots([2,2,2,3,5], field=GF)
    b = galois.Poly.Roots([1,2,6], field=GF)
    gcd, x, y = galois.poly_gcd(a, b)
    assert a*x + b*y == gcd

    GF = galois.GF(2**5, irreducible_poly=galois.Poly([1, 0, 1, 0, 0, 1], order="asc"))
    a = galois.Poly([1, 2, 4, 1], field=GF)
    b = galois.Poly([9, 3], field=GF)
    gcd, x, y = galois.poly_gcd(a, b)
    assert a*x + b*y == gcd

def test_poly_gcd_unit():
    GF = galois.GF(7)
    a = galois.conway_poly(7, 10)
    b = galois.conway_poly(7, 5)
    gcd = galois.poly_gcd(a, b)[0]
    assert gcd == galois.Poly([1], field=GF)  # Note, not 3


def test_poly_pow():
    GF = galois.GF(31)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 20
    assert f**power % g == galois.poly_pow(f, power, g)


def test_is_irreducible():
    p1 = galois.conway_poly(2, 4)
    assert galois.is_irreducible(p1)

    p2 = galois.conway_poly(2, 5)
    assert galois.is_irreducible(p2)

    p3 = p1 * p2  # Has no roots in GF(2) but is still reducible
    assert not galois.is_irreducible(p3)

    p4 = galois.conway_poly(7, 20)
    assert galois.is_irreducible(p4)

    GF = galois.GF(7)
    p5 = galois.Poly.Roots([2, 3], field=GF)
    assert not galois.is_irreducible(p5)

    GF = galois.GF(7)
    x = galois.Poly.Identity(GF)
    p6 = galois.Poly.Random(20, field=GF) * x
    assert not galois.is_irreducible(p6)


def test_is_primitive():
    assert galois.is_primitive(galois.conway_poly(2, 2))
    assert galois.is_primitive(galois.conway_poly(2, 3))
    assert galois.is_primitive(galois.conway_poly(2, 4))
    assert galois.is_primitive(galois.conway_poly(2, 5))
    assert galois.is_primitive(galois.conway_poly(2, 6))
    assert galois.is_primitive(galois.conway_poly(2, 7))
    assert galois.is_primitive(galois.conway_poly(2, 8))

    # The AES irreducible polynomial is not primitive
    p = galois.Poly.Degrees([8,4,3,1,0])
    assert not galois.is_primitive(p)

    assert galois.is_primitive(galois.conway_poly(3, 2))
    assert galois.is_primitive(galois.conway_poly(3, 3))
    assert galois.is_primitive(galois.conway_poly(3, 4))
    assert galois.is_primitive(galois.conway_poly(3, 5))


def test_is_monic():
    GF = galois.GF(7)
    p = galois.Poly([1,0,4,5], field=GF)
    assert galois.is_monic(p)

    GF = galois.GF(7)
    p = galois.Poly([3,0,4,5], field=GF)
    assert not galois.is_monic(p)
