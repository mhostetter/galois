"""
A pytest module to test general functions on polynomials over finite fields.
"""
import pytest

import galois


def test_is_monic_exceptions():
    GF = galois.GF(7)
    p = galois.Poly([1,0,4,5], field=GF)

    with pytest.raises(TypeError):
        galois.is_monic(p.coeffs)


def test_is_monic():
    GF = galois.GF(7)
    p = galois.Poly([1,0,4,5], field=GF)
    assert galois.is_monic(p)

    GF = galois.GF(7)
    p = galois.Poly([3,0,4,5], field=GF)
    assert not galois.is_monic(p)


def test_is_irreducible_exceptions():
    with pytest.raises(TypeError):
        galois.is_irreducible([1, 0, 1, 1])


def test_is_irreducible(poly_is_irreducible):
    GF, IS, IS_NOT = poly_is_irreducible["GF"], poly_is_irreducible["IS"], poly_is_irreducible["IS_NOT"]
    for i in range(len(IS)):
        p = IS[i]
        assert galois.is_irreducible(p)
        p = IS_NOT[i]
        assert not galois.is_irreducible(p)


def test_is_primitive_exceptions():
    with pytest.raises(TypeError):
        galois.is_primitive([1, 0, 1, 1])


def test_is_primitive(poly_is_primitive):
    GF, IS, IS_NOT = poly_is_primitive["GF"], poly_is_primitive["IS"], poly_is_primitive["IS_NOT"]
    for i in range(len(IS)):
        p = IS[i]
        assert galois.is_primitive(p)
        p = IS_NOT[i]
        assert not galois.is_primitive(p)


def test_is_square_free_exceptions():
    with pytest.raises(TypeError):
        galois.is_square_free([1, 0, 1, 1])


def test_is_square_free(poly_is_square_free):
    GF, X, Z = poly_is_square_free["GF"], poly_is_square_free["X"], poly_is_square_free["Z"]
    for i in range(len(X)):
        assert galois.is_square_free(X[i]) == Z[i]
