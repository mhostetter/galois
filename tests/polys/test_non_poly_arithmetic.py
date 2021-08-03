"""
A pytest module to test polynomial and non-polynomial arithmetic.
"""
import pytest
import numpy as np

import galois


def test_add():
    GF = galois.GF(3)
    poly = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(e)
    assert poly + e == poly + e_poly
    assert e + poly == e_poly + poly

    # Not a Galois field array
    with pytest.raises(TypeError):
        poly + 1
    with pytest.raises(TypeError):
        1 + poly

    # Not a 0-D Galois field array
    with pytest.raises(ValueError):
        poly + GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) + poly


def test_subtract():
    GF = galois.GF(3)
    poly = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(e)
    assert poly - e == poly - e_poly
    assert e - poly == e_poly - poly

    # Not a Galois field array
    with pytest.raises(TypeError):
        poly - 1
    with pytest.raises(TypeError):
        1 - poly

    # Not a 0-D Galois field array
    with pytest.raises(ValueError):
        poly - GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) - poly


def test_multiply():
    GF = galois.GF(3)
    poly = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(e)
    assert poly * e == poly * e_poly
    assert e * poly == e_poly * poly

    # Not a Galois field array
    with pytest.raises(TypeError):
        poly * 1
    with pytest.raises(TypeError):
        1 * poly

    # Not a 0-D Galois field array
    with pytest.raises(ValueError):
        poly * GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) * poly


def test_true_divide():
    GF = galois.GF(3)
    poly = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(e)
    assert poly / e == poly / e_poly
    assert e / poly == e_poly / poly

    # Not a Galois field array
    with pytest.raises(TypeError):
        poly / 1
    with pytest.raises(TypeError):
        1 / poly

    # Not a 0-D Galois field array
    with pytest.raises(ValueError):
        poly / GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) / poly


def test_floor_divide():
    GF = galois.GF(3)
    poly = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(e)
    assert poly // e == poly // e_poly
    assert e // poly == e_poly // poly

    # Not a Galois field array
    with pytest.raises(TypeError):
        poly // 1
    with pytest.raises(TypeError):
        1 // poly

    # Not a 0-D Galois field array
    with pytest.raises(ValueError):
        poly // GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) // poly


def test_mod():
    GF = galois.GF(3)
    poly = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(e)
    assert poly % e == poly % e_poly
    assert e % poly == e_poly % poly

    # Not a Galois field array
    with pytest.raises(TypeError):
        poly % 1
    with pytest.raises(TypeError):
        1 % poly

    # Not a 0-D Galois field array
    with pytest.raises(ValueError):
        poly % GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) % poly


def test_equal():
    GF = galois.GF(3)
    poly = galois.Poly([2], field=GF)
    assert poly == galois.Poly([2], field=GF)

    # Not a polynomial
    with pytest.raises(TypeError):
        assert poly == GF(2)
    with pytest.raises(TypeError):
        poly == 2
