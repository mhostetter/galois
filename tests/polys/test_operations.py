"""
A pytest module to test various Galois field polynomial operations.
"""

import numpy as np
import pytest

import galois


def test_coefficients_exceptions():
    GF = galois.GF(7)
    p = galois.Poly([3, 0, 5, 2], field=GF)

    with pytest.raises(TypeError):
        p.coefficients(8.0)
    with pytest.raises(TypeError):
        p.coefficients(order=1)
    with pytest.raises(ValueError):
        p.coefficients(3)
    with pytest.raises(ValueError):
        p.coefficients(order="ascending")


def test_coefficients():
    GF = galois.GF(7)
    p = galois.Poly([3, 0, 5, 2], field=GF)
    assert np.array_equal(p.coefficients(), p.coeffs)

    coeffs = p.coefficients()
    assert np.array_equal(coeffs, [3, 0, 5, 2])
    assert type(coeffs) is GF

    coeffs = p.coefficients(order="asc")
    assert np.array_equal(coeffs, [2, 5, 0, 3])
    assert type(coeffs) is GF

    coeffs = p.coefficients(6)
    assert np.array_equal(coeffs, [0, 0, 3, 0, 5, 2])
    assert type(coeffs) is GF

    coeffs = p.coefficients(6, order="asc")
    assert np.array_equal(coeffs, [2, 5, 0, 3, 0, 0])
    assert type(coeffs) is GF


def test_reverse():
    GF = galois.GF(3)
    p1 = galois.Poly([2, 0, 1, 2], field=GF)
    p2 = galois.Poly([2, 1, 0, 2], field=GF)
    assert p1._type == p2._type == "dense"
    assert p1.reverse() == p2

    p1 = galois.Poly([1, 0, 1, 1])
    p2 = galois.Poly([1, 1, 0, 1])
    assert p1._type == p2._type == "binary"
    assert p1.reverse() == p2

    GF = galois.GF(3)
    p1 = galois.Poly.Degrees([3000, 1, 0], [1, 2, 1], field=GF)
    p2 = galois.Poly.Degrees([3000, 2999, 0], [1, 2, 1], field=GF)
    assert p1._type == p2._type == "sparse"
    assert p1.reverse() == p2


def test_repr():
    GF = galois.GF2
    poly = galois.Poly([1, 0, 1, 1])
    assert repr(poly) == "Poly(x^3 + x + 1, GF(2))"
    with GF.repr("poly"):
        assert repr(poly) == "Poly(x^3 + x + 1, GF(2))"
    with GF.repr("power"):
        assert repr(poly) == "Poly(x^3 + x + 1, GF(2))"

    GF = galois.GF(7)
    poly = galois.Poly([5, 0, 3, 1], field=GF)
    assert repr(poly) == "Poly(5x^3 + 3x + 1, GF(7))"
    with GF.repr("poly"):
        assert repr(poly) == "Poly((5)x^3 + (3)x + 1, GF(7))"
    with GF.repr("power"):
        assert repr(poly) == "Poly((α^5)x^3 + (α)x + 1, GF(7))"

    GF = galois.GF(2**3)
    poly = galois.Poly([2, 0, 3, 1], field=GF)
    assert repr(poly) == "Poly(2x^3 + 3x + 1, GF(2^3))"
    with GF.repr("poly"):
        assert repr(poly) == "Poly((α)x^3 + (α + 1)x + 1, GF(2^3))"
    with GF.repr("power"):
        assert repr(poly) == "Poly((α)x^3 + (α^3)x + 1, GF(2^3))"


def test_str():
    GF = galois.GF2
    poly = galois.Poly([1, 0, 1, 1])
    assert str(poly) == "x^3 + x + 1"
    with GF.repr("poly"):
        assert str(poly) == "x^3 + x + 1"
    with GF.repr("power"):
        assert str(poly) == "x^3 + x + 1"

    GF = galois.GF(7)
    poly = galois.Poly([5, 0, 3, 1], field=GF)
    assert str(poly) == "5x^3 + 3x + 1"
    with GF.repr("poly"):
        assert str(poly) == "(5)x^3 + (3)x + 1"
    with GF.repr("power"):
        assert str(poly) == "(α^5)x^3 + (α)x + 1"

    GF = galois.GF(2**3)
    poly = galois.Poly([2, 0, 3, 5], field=GF)
    assert str(poly) == "2x^3 + 3x + 5"
    with GF.repr("poly"):
        assert str(poly) == "(α)x^3 + (α + 1)x + (α^2 + 1)"
    with GF.repr("power"):
        assert str(poly) == "(α)x^3 + (α^3)x + α^6"


def test_int():
    poly = galois.Poly(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
    )
    assert int(poly) == 4295000729

    poly = galois.Poly.Degrees([32, 15, 9, 7, 4, 3, 0])
    assert int(poly) == 4295000729


def test_bin():
    poly = galois.Poly([1, 0, 1, 1])
    assert bin(poly) == "0b1011"


def test_oct():
    GF = galois.GF(2**3)
    poly = galois.Poly([5, 0, 3, 4], field=GF)
    assert oct(poly) == "0o5034"


def test_hex():
    GF = galois.GF(2**8)
    poly = galois.Poly([0xF7, 0x00, 0xA2, 0x75], field=GF)
    assert hex(poly) == "0xf700a275"


def test_equal(field):
    c = field.Random(6)
    c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    p1 = galois.Poly(c)
    p2 = galois.Poly(c.tolist(), field=field)
    assert p1 == p2
    assert p2 == p1


def test_equal_coeffs_diff_field(field):
    # NOTE: GF(11) is not included in the `field` pytest fixture
    c = field.Ones(6)
    p1 = galois.Poly(c)
    p2 = galois.Poly(c.tolist(), field=galois.GF(11))
    assert p1 != p2
    assert p2 != p1


def test_cant_set_coeffs():
    GF = galois.GF(7)
    coeffs = [5, 0, 0, 4, 2, 0, 3]
    p = galois.Poly(coeffs, field=GF)
    assert np.array_equal(p.coeffs, coeffs)
    p.coeffs[-1] = 0
    assert np.array_equal(p.coeffs, coeffs)

    coeffs = [1, 0, 0, 1, 1, 0, 1]
    p = galois.Poly(coeffs)
    assert np.array_equal(p.coeffs, coeffs)
    p.coeffs[-1] = 0
    assert np.array_equal(p.coeffs, coeffs)


def test_len():
    GF = galois.GF(7)
    coeffs = [1, 0, 0, 0, 0]
    p = galois.Poly(coeffs, field=GF)
    assert p._type == "dense"
    assert len(p) == 5
    p = galois.Poly(coeffs[::-1], field=GF)
    assert p._type == "dense"
    assert len(p) == 1
    p = galois.Poly([0, 0, 0], field=GF)
    assert p._type == "dense"
    assert len(p) == 1

    coeffs = [1, 0, 0, 0, 0]
    p = galois.Poly(coeffs)
    assert p._type == "binary"
    assert len(p) == 5
    p = galois.Poly(coeffs[::-1])
    assert p._type == "binary"
    assert len(p) == 1
    p = galois.Poly([0, 0, 0])
    assert p._type == "binary"
    assert len(p) == 1

    p = galois.Poly.Str("x^2000 + 1", field=GF)
    assert p._type == "sparse"
    assert len(p) == 2001


def test_immutable():
    GF = galois.GF(7)
    f = galois.Poly([1, 2, 3], field=GF)
    h = f
    f += GF(1)
    assert f == galois.Poly([1, 2, 4], field=GF)
    assert h == galois.Poly([1, 2, 3], field=GF)
