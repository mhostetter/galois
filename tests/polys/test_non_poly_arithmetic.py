"""
A pytest module to test polynomial and non-polynomial arithmetic.
"""
import random

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

    # Not a 0-D Galois field array
    with pytest.raises(ValueError):
        poly * GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) * poly


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


def test_equal_with_scalar(field):
    # NOTE: GF(11) is not included in the `field` pytest fixture
    scalar = 0
    p = galois.Poly([scalar], field=field)
    assert p == scalar
    assert p == field(scalar)
    assert p == galois.Poly([scalar], field=field)
    assert p != galois.Poly([scalar], field=galois.GF(11))
    assert scalar == p
    assert field(scalar) == p
    assert galois.Poly([scalar], field=field) == p
    assert galois.Poly([scalar], field=galois.GF(11)) != p

    scalar = 1
    p = galois.Poly([scalar], field=field)
    assert p == scalar
    assert p == field(scalar)
    assert p == galois.Poly([scalar], field=field)
    assert p != galois.Poly([scalar], field=galois.GF(11))
    assert scalar == p
    assert field(scalar) == p
    assert galois.Poly([scalar], field=field) == p
    assert galois.Poly([scalar], field=galois.GF(11)) != p

    scalar = random.randint(1, field.order - 1)
    p = galois.Poly([scalar], field=field)
    assert p == scalar
    assert p == field(scalar)
    assert p == galois.Poly([scalar], field=field)
    assert scalar == p
    assert field(scalar) == p
    assert galois.Poly([scalar], field=field) == p


def test_equal_with_vector(field):
    # NOTE: GF(11) is not included in the `field` pytest fixture
    c = field.Random(6)
    c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    with pytest.raises(ValueError):
        # Comparing polynomials against FieldArrays (non-scalars) isn't supported
        galois.Poly(c) == c

    c = field.Ones(6)
    with pytest.raises(ValueError):
        # Comparing polynomials against FieldArrays (non-scalars) isn't supported
        galois.Poly(c) == c
