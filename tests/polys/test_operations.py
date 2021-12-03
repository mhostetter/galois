"""
A pytest module to test various Galois field polynomial operations.
"""
import numpy as np

import galois
from galois._polys._poly import DensePoly, BinaryPoly, SparsePoly


def test_copy():
    p1 = galois.Poly([2,0,1,2], field=galois.GF(3))
    assert isinstance(p1, DensePoly)
    p2 = p1.copy()
    p2._coeffs[0] = 0
    assert np.array_equal(p1.coeffs, [2,0,1,2])

    p1 = galois.Poly([1,0,1,1])
    assert isinstance(p1, BinaryPoly)
    p2 = p1.copy()
    p2._integer = 27
    assert np.array_equal(p1.coeffs, [1,0,1,1])

    p1 = galois.Poly.Degrees([3000,1,0], [1,2,1], field=galois.GF(3))
    assert isinstance(p1, SparsePoly)
    p2 = p1.copy()
    p2._degrees[0] = 4000
    assert np.array_equal(p1.nonzero_degrees, [3000,1,0])


def test_reverse():
    p1 = galois.Poly([2,0,1,2], field=galois.GF(3))
    p2 = galois.Poly([2,1,0,2], field=galois.GF(3))
    assert p1.reverse() == p2

    p1 = galois.Poly([1,0,1,1])
    p2 = galois.Poly([1,1,0,1])
    assert p1.reverse() == p2

    p1 = galois.Poly.Degrees([3000,1,0], [1,2,1], field=galois.GF(3))
    p2 = galois.Poly.Degrees([3000,2999,0], [1,2,1], field=galois.GF(3))
    assert p1.reverse() == p2


def test_string():
    GF = galois.GF2
    poly = galois.Poly([1,0,1,1])
    assert repr(poly) == str(poly) == "Poly(x^3 + x + 1, GF(2))"
    with GF.display("poly"):
        assert repr(poly) == str(poly) == "Poly(x^3 + x + (1), GF(2))"  # TODO: Clean this up
    with GF.display("power"):
        assert repr(poly) == str(poly) == "Poly(x^3 + x + (1), GF(2))"  # TODO: Clean this up

    GF = galois.GF(7)
    poly = galois.Poly([5,0,3,1], field=GF)
    assert repr(poly) == str(poly) == "Poly(5x^3 + 3x + 1, GF(7))"
    with GF.display("poly"):
        assert repr(poly) == str(poly) == "Poly((5)x^3 + (3)x + (1), GF(7))"  # TODO: Clean this up
    with GF.display("power"):
        assert repr(poly) == str(poly) == "Poly((α^5)x^3 + (α)x + (1), GF(7))"

    GF = galois.GF(2**3)
    poly = galois.Poly([2,0,3,1], field=GF)
    assert repr(poly) == str(poly) == "Poly(2x^3 + 3x + 1, GF(2^3))"
    with GF.display("poly"):
        assert repr(poly) == str(poly) == "Poly((α)x^3 + (α + 1)x + (1), GF(2^3))"
    with GF.display("power"):
        assert repr(poly) == str(poly) == "Poly((α)x^3 + (α^3)x + (1), GF(2^3))"


def test_integer():
    poly = galois.Poly([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,0,0,1])
    assert poly.integer == 4295000729

    poly = galois.Poly.Degrees([32,15,9,7,4,3,0])
    assert poly.integer == 4295000729


def test_equal(field):
    c = field.Random(6)
    c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    p1 = galois.Poly(c)
    p2 = galois.Poly(c.tolist(), field=field)
    assert p1 == p2


def test_cant_set_coeffs():
    # DensePoly
    GF = galois.GF(7)
    coeffs = [5,0,0,4,2,0,3]
    p = galois.Poly(coeffs, field=GF)
    assert np.array_equal(p.coeffs, coeffs)
    p.coeffs[-1] = 0
    assert np.array_equal(p.coeffs, coeffs)

    # BinaryPoly
    coeffs = [1,0,0,1,1,0,1]
    p = galois.Poly(coeffs)
    assert np.array_equal(p.coeffs, coeffs)
    p.coeffs[-1] = 0
    assert np.array_equal(p.coeffs, coeffs)


def test_len():
    # DensePoly
    GF = galois.GF(7)
    coeffs = [5, 0, 0, 4, 2, 0, 3]
    p = galois.Poly(coeffs, field=GF)
    assert len(p) == 7

    # BinaryPoly
    coeffs = [1, 0, 0, 0, 0]
    p = galois.Poly(coeffs)
    assert len(p) == 5
    p = galois.Poly(coeffs[::-1])
    assert len(p) == 1
    p = galois.Poly([0, 0, 0])
    assert len(p) == 1

    # SparsePoly
    p = galois.Poly.String("x^1000 + 1")
    assert len(p) == 1001
