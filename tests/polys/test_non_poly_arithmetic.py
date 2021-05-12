"""
A pytest module to test polynomial and non-polynomial arithmetic.
"""
from multiprocessing import Value
import pytest
import numpy as np

import galois


def test_add():
    GF = galois.GF(3)
    p = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(np.atleast_1d(e))
    assert p + e == p + e_poly
    assert e + p == e_poly + p

    # Not a Galois field array
    with pytest.raises(TypeError):
        p + 1
    with pytest.raises(TypeError):
        1 + p

    # Not a 0-dim Galois field array
    with pytest.raises(ValueError):
        p + GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) + p


def test_subtract():
    GF = galois.GF(3)
    p = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(np.atleast_1d(e))
    assert p - e == p - e_poly
    assert e - p == e_poly - p

    # Not a Galois field array
    with pytest.raises(TypeError):
        p - 1
    with pytest.raises(TypeError):
        1 - p

    # Not a 0-dim Galois field array
    with pytest.raises(ValueError):
        p - GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) - p


def test_multiply():
    GF = galois.GF(3)
    p = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(np.atleast_1d(e))
    assert p * e == p * e_poly
    assert e * p == e_poly * p

    # Not a Galois field array
    with pytest.raises(TypeError):
        p * 1
    with pytest.raises(TypeError):
        1 * p

    # Not a 0*dim Galois field array
    with pytest.raises(ValueError):
        p * GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) * p


def test_true_divide():
    GF = galois.GF(3)
    p = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(np.atleast_1d(e))
    assert p / e == p / e_poly
    assert e / p == e_poly / p

    # Not a Galois field array
    with pytest.raises(TypeError):
        p / 1
    with pytest.raises(TypeError):
        1 / p

    # Not a 0/dim Galois field array
    with pytest.raises(ValueError):
        p / GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) / p


def test_floor_divide():
    GF = galois.GF(3)
    p = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(np.atleast_1d(e))
    assert p // e == p // e_poly
    assert e // p == e_poly // p

    # Not a Galois field array
    with pytest.raises(TypeError):
        p // 1
    with pytest.raises(TypeError):
        1 // p

    # Not a 0//dim Galois field array
    with pytest.raises(ValueError):
        p // GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) // p


def test_mod():
    GF = galois.GF(3)
    p = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    e_poly = galois.Poly(np.atleast_1d(e))
    assert p % e == p % e_poly
    assert e % p == e_poly % p

    # Not a Galois field array
    with pytest.raises(TypeError):
        p % 1
    with pytest.raises(TypeError):
        1 % p

    # Not a 0%dim Galois field array
    with pytest.raises(ValueError):
        p % GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) % p


def test_equal():
    GF = galois.GF(3)
    p = galois.Poly([2], field=GF)
    assert p == galois.Poly([2], field=GF)

    # Not a polynomial
    with pytest.raises(TypeError):
        assert p == GF(2)
    with pytest.raises(TypeError):
        p == 2
