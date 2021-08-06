"""
A pytest module to test functions on Galois field polynomials.
"""
import random

import pytest

import galois


def test_poly_factors():
    GF = galois.GF2
    g0, g1, g2 = galois.conway_poly(2, 3), galois.conway_poly(2, 4), galois.conway_poly(2, 5)
    k0, k1, k2 = 2, 3, 4
    f = g0**k0 * g1**k1 * g2**k2
    factors, multiplicities = galois.poly_factors(f)
    assert factors == [g0, g1, g2]
    assert multiplicities == [k0, k1, k2]

    GF = galois.GF(3)
    g0, g1, g2 = galois.conway_poly(3, 3), galois.conway_poly(3, 4), galois.conway_poly(3, 5)
    g0, g1, g2
    k0, k1, k2 = 3, 4, 6
    f = g0**k0 * g1**k1 * g2**k2
    factors, multiplicities = galois.poly_factors(f)
    assert factors == [g0, g1, g2]
    assert multiplicities == [k0, k1, k2]


def test_is_monic():
    GF = galois.GF(7)
    p = galois.Poly([1,0,4,5], field=GF)
    assert galois.is_monic(p)

    GF = galois.GF(7)
    p = galois.Poly([3,0,4,5], field=GF)
    assert not galois.is_monic(p)


def test_is_monic_exceptions():
    GF = galois.GF(7)
    p = galois.Poly([1,0,4,5], field=GF)

    with pytest.raises(TypeError):
        galois.is_monic(p.coeffs)
