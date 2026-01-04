"""
A pytest module to test polynomial test (is_blank) method and properties.
"""

import galois


def test_is_zero():
    assert galois.Poly.Zero().is_zero
    assert not galois.Poly.One().is_zero


def test_is_one():
    assert galois.Poly.One().is_one
    assert not galois.Poly.Identity().is_one


def test_is_monic(poly_is_monic):
    X, Z = poly_is_monic["X"], poly_is_monic["Z"]
    for x, z in zip(X, Z):
        assert x.is_monic == z


def test_is_irreducible(poly_is_irreducible):
    IS, IS_NOT = poly_is_irreducible["IS"], poly_is_irreducible["IS_NOT"]
    for p_is, p_is_not in zip(IS, IS_NOT):
        assert p_is.is_irreducible()
        assert not p_is_not.is_irreducible()


def test_is_primitive(poly_is_primitive):
    IS, IS_NOT = poly_is_primitive["IS"], poly_is_primitive["IS_NOT"]
    for p_is, p_is_not in zip(IS, IS_NOT):
        assert p_is.is_primitive()
        assert not p_is_not.is_primitive()


def test_is_square_free(poly_is_square_free):
    X, Z = poly_is_square_free["X"], poly_is_square_free["Z"]
    for x, z in zip(X, Z):
        assert x.is_square_free() == z
