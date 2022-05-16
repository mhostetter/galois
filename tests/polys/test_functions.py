"""
A pytest module to test general functions on polynomials over finite fields.
"""
import pytest

import galois


def test_is_monic(poly_is_monic):
    X, Z = poly_is_monic["X"], poly_is_monic["Z"]
    for i in range(len(X)):
        assert X[i].is_monic == Z[i]


def test_is_irreducible(poly_is_irreducible):
    IS, IS_NOT = poly_is_irreducible["IS"], poly_is_irreducible["IS_NOT"]
    for i in range(len(IS)):
        p = IS[i]
        assert p.is_irreducible()
        p = IS_NOT[i]
        assert not p.is_irreducible()


def test_is_primitive(poly_is_primitive):
    IS, IS_NOT = poly_is_primitive["IS"], poly_is_primitive["IS_NOT"]
    for i in range(len(IS)):
        p = IS[i]
        assert p.is_primitive()
        p = IS_NOT[i]
        assert not p.is_primitive()


def test_is_square_free_exceptions():
    with pytest.raises(TypeError):
        galois.is_square_free([1, 0, 1, 1])


def test_is_square_free(poly_is_square_free):
    X, Z = poly_is_square_free["X"], poly_is_square_free["Z"]
    for i in range(len(X)):
        assert galois.is_square_free(X[i]) == Z[i]
