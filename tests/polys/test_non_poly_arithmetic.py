"""
A pytest module to test polynomial and non-polynomial arithmetic.
"""

import pytest

import galois


def test_add():
    GF = galois.GF(3)
    f = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    g_e = galois.Poly(e)
    assert f + e == f + g_e
    assert e + f == g_e + f

    # Not a FieldArray
    with pytest.raises(TypeError):
        f + 1
    with pytest.raises(TypeError):
        1 + f

    # Not a 0-D FieldArray
    with pytest.raises(ValueError):
        f + GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) + f


def test_subtract():
    GF = galois.GF(3)
    f = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    g_e = galois.Poly(e)
    assert f - e == f - g_e
    assert e - f == g_e - f

    # Not a FieldArray
    with pytest.raises(TypeError):
        f - 1
    with pytest.raises(TypeError):
        1 - f

    # Not a 0-D FieldArray
    with pytest.raises(ValueError):
        f - GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) - f


def test_multiply():
    GF = galois.GF(3)
    f = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    g_e = galois.Poly(e)
    assert f * e == f * g_e
    assert e * f == g_e * f

    # Not a 0-D FieldArray
    with pytest.raises(ValueError):
        f * GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) * f


def test_floor_divide():
    GF = galois.GF(3)
    f = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    g_e = galois.Poly(e)
    assert f // e == f // g_e
    assert e // f == g_e // f

    # Not a FieldArray
    with pytest.raises(TypeError):
        f // 1
    with pytest.raises(TypeError):
        1 // f

    # Not a 0-D FieldArray
    with pytest.raises(ValueError):
        f // GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) // f


def test_mod():
    GF = galois.GF(3)
    f = galois.Poly.Random(5, field=GF)
    e = GF.Random(low=1)  # Random field element
    g_e = galois.Poly(e)
    assert f % e == f % g_e
    assert e % f == g_e % f

    # Not a FieldArray
    with pytest.raises(TypeError):
        f % 1
    with pytest.raises(TypeError):
        1 % f

    # Not a 0-D FieldArray
    with pytest.raises(ValueError):
        f % GF.Random(3, low=1)
    with pytest.raises(ValueError):
        GF.Random(3, low=1) % f


def test_equal_with_poly_like():
    GF = galois.GF(7)
    f = galois.Poly([3, 0, 2, 5], field=GF)
    assert f == galois.Poly([3, 0, 2, 5], field=GF)
    assert f != galois.Poly([3, 0, 2, 5], field=galois.GF(11))
    assert f == GF([3, 0, 2, 5])
    assert f != galois.GF(11)([3, 0, 2, 5])
    assert f == [3, 0, 2, 5]
    assert f == "3x^3 + 2x + 5"
    assert f == "3x**3+2x+5"
    assert f == 1048
