"""
A pytest module to test functions on Galois field polynomials.
"""
import random

import pytest

import galois


def test_poly_gcd():
    a = galois.Poly.Random(5)
    b = galois.Poly.Zero()
    gcd = galois.poly_gcd(a, b)
    assert gcd == a

    a = galois.Poly.Zero()
    b = galois.Poly.Random(5)
    gcd = galois.poly_gcd(a, b)
    assert gcd == b

    # Example 2.223 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])
    gcd = galois.poly_gcd(a, b)
    assert gcd == galois.Poly.Degrees([3, 1, 0])

    # Tested against SageMath
    GF = galois.GF(3)
    a = galois.Poly.Degrees([11, 9, 8, 6, 5, 3, 2, 0], [1, 2, 2, 1, 1, 2, 2, 1], field=GF)
    b = a.derivative()
    gcd = galois.poly_gcd(a, b)
    assert gcd == galois.Poly.Degrees([9, 6, 3, 0], [1, 2, 1, 2], field=GF)


def test_poly_gcd_unit():
    GF = galois.GF(7)
    a = galois.conway_poly(7, 10)
    b = galois.conway_poly(7, 5)
    gcd = galois.poly_gcd(a, b)
    assert gcd == galois.Poly([1], field=GF)  # Note, not 3


def test_poly_gcd_exceptions():
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])

    with pytest.raises(TypeError):
        galois.poly_gcd(a.coeffs, b)
    with pytest.raises(TypeError):
        galois.poly_gcd(a, b.coeffs)
    with pytest.raises(ValueError):
        galois.poly_gcd(a, galois.Poly(b.coeffs, field=galois.GF(3)))


def test_poly_egcd():
    a = galois.Poly.Random(5)
    b = galois.Poly.Zero()
    gcd, x, y = galois.poly_egcd(a, b)
    assert gcd == a
    assert a*x + b*y == gcd

    a = galois.Poly.Zero()
    b = galois.Poly.Random(5)
    gcd, x, y = galois.poly_egcd(a, b)
    assert gcd == b
    assert a*x + b*y == gcd

    # Example 2.223 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])
    gcd, x, y = galois.poly_egcd(a, b)
    assert gcd == galois.Poly.Degrees([3, 1, 0])
    assert x == galois.Poly.Degrees([4])
    assert y == galois.Poly.Degrees([5, 4, 3, 2, 1, 0])
    assert a*x + b*y == gcd

    # Tested against SageMath
    GF = galois.GF(3)
    a = galois.Poly.Degrees([11, 9, 8, 6, 5, 3, 2, 0], [1, 2, 2, 1, 1, 2, 2, 1], field=GF)
    b = a.derivative()
    gcd, x, y = galois.poly_egcd(a, b)
    assert gcd == galois.Poly.Degrees([9, 6, 3, 0], [1, 2, 1, 2], field=GF)
    assert x == galois.Poly([2], field=GF)
    assert y == galois.Poly([2, 0], field=GF)
    assert a*x + b*y == gcd

    GF = galois.GF(7)
    a = galois.Poly.Roots([2,2,2,3,5], field=GF)
    b = galois.Poly.Roots([1,2,6], field=GF)
    gcd, x, y = galois.poly_egcd(a, b)
    assert a*x + b*y == gcd

    GF = galois.GF(2**5, irreducible_poly=galois.Poly([1, 0, 1, 0, 0, 1], order="asc"))
    a = galois.Poly([1, 2, 4, 1], field=GF)
    b = galois.Poly([9, 3], field=GF)
    gcd, x, y = galois.poly_egcd(a, b)
    assert a*x + b*y == gcd

def test_poly_egcd_unit():
    GF = galois.GF(7)
    a = galois.conway_poly(7, 10)
    b = galois.conway_poly(7, 5)
    gcd, _, _ = galois.poly_egcd(a, b)
    assert gcd == galois.Poly([1], field=GF)  # Note, not 3


def test_poly_egcd_exceptions():
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])

    with pytest.raises(TypeError):
        galois.poly_egcd(a.coeffs, b)
    with pytest.raises(TypeError):
        galois.poly_egcd(a, b.coeffs)
    with pytest.raises(ValueError):
        galois.poly_egcd(a, galois.Poly(b.coeffs, field=galois.GF(3)))


def test_poly_gcd_vs_egcd():
    GF = galois.GF2
    a = galois.Poly.Random(random.randint(5, 20), field=GF)
    b = galois.Poly.Random(random.randint(5, 20), field=GF)
    assert galois.poly_gcd(a, b) == galois.poly_egcd(a, b)[0]

    GF = galois.GF(7)
    a = galois.Poly.Random(random.randint(5, 20), field=GF)
    b = galois.Poly.Random(random.randint(5, 20), field=GF)
    assert galois.poly_gcd(a, b) == galois.poly_egcd(a, b)[0]

    GF = galois.GF(2**8)
    a = galois.Poly.Random(random.randint(5, 20), field=GF)
    b = galois.Poly.Random(random.randint(5, 20), field=GF)
    assert galois.poly_gcd(a, b) == galois.poly_egcd(a, b)[0]

    GF = galois.GF(7**3)
    a = galois.Poly.Random(random.randint(5, 20), field=GF)
    b = galois.Poly.Random(random.randint(5, 20), field=GF)
    assert galois.poly_gcd(a, b) == galois.poly_egcd(a, b)[0]


def test_poly_pow():
    GF = galois.GF(31)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 20
    assert f**power % g == galois.poly_pow(f, power, g)

    GF = galois.GF(31)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 0
    assert f**power % g == galois.poly_pow(f, power, g)


def test_poly_pow_exceptions():
    GF = galois.GF(31)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 20

    with pytest.raises(TypeError):
        galois.poly_pow(f.coeffs, power, g)
    with pytest.raises(TypeError):
        galois.poly_pow(f, float(power), g)
    with pytest.raises(TypeError):
        galois.poly_pow(f, power, g.coeffs)
    with pytest.raises(ValueError):
        galois.poly_pow(f, -power, g)


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


def test_irreducible_poly():
    p = galois.irreducible_poly(2, 3)
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    p = galois.irreducible_poly(2, 3, method="max")
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    p = galois.irreducible_poly(2, 3, method="random")
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)

    p = galois.irreducible_poly(2, 8)
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    p = galois.irreducible_poly(2, 8, method="max")
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    p = galois.irreducible_poly(2, 8, method="random")
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)

    p = galois.irreducible_poly(3, 5)
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    p = galois.irreducible_poly(3, 5, method="max")
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    p = galois.irreducible_poly(3, 5, method="random")
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)


def test_irreducible_polys():
    """
    Generated with Octave:
        primpoly(5, 'all')
        primpoly(6, 'all')
        primpoly(7, 'all')
        primpoly(8, 'all')
    """
    assert set([p.integer for p in galois.irreducible_polys(2, 1)]).issuperset(set([3]))
    assert set([p.integer for p in galois.irreducible_polys(2, 2)]).issuperset(set([7]))
    assert set([p.integer for p in galois.irreducible_polys(2, 3)]).issuperset(set([11, 13]))
    assert set([p.integer for p in galois.irreducible_polys(2, 4)]).issuperset(set([19, 25]))
    assert set([p.integer for p in galois.irreducible_polys(2, 5)]).issuperset(set([37, 41, 47, 55, 59, 61]))
    assert set([p.integer for p in galois.irreducible_polys(2, 6)]).issuperset(set([67, 91, 97, 103, 109, 115]))
    assert set([p.integer for p in galois.irreducible_polys(2, 7)]).issuperset(set([131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213, 229, 239, 241, 247, 253]))
    assert set([p.integer for p in galois.irreducible_polys(2, 8)]).issuperset(set([285, 299, 301, 333, 351, 355, 357, 361, 369, 391, 397, 425, 451, 463, 487, 501]))

    # https://oeis.org/A001037
    # assert len(galois.irreducible_polys(2, 0)) == 1
    assert len(galois.irreducible_polys(2, 1)) == 2
    assert len(galois.irreducible_polys(2, 2)) == 1
    assert len(galois.irreducible_polys(2, 3)) == 2
    assert len(galois.irreducible_polys(2, 4)) == 3
    assert len(galois.irreducible_polys(2, 5)) == 6
    assert len(galois.irreducible_polys(2, 6)) == 9
    assert len(galois.irreducible_polys(2, 7)) == 18
    assert len(galois.irreducible_polys(2, 8)) == 30
    assert len(galois.irreducible_polys(2, 9)) == 56
    assert len(galois.irreducible_polys(2, 10)) == 99


def test_irreducible_polys_product():
    GF = galois.GF2
    polys_1 = galois.irreducible_polys(2, 1)
    polys_2 = galois.irreducible_polys(2, 2)
    polys_4 = galois.irreducible_polys(2, 4)
    g = galois.Poly.One(GF)
    for p in polys_1 + polys_2 + polys_4:
        g *= p
    x = galois.Poly.Identity(GF)
    assert g == x**(2**4) - x

    GF = galois.GF(3)
    polys_1 = galois.irreducible_polys(3, 1)
    polys_2 = galois.irreducible_polys(3, 2)
    polys_4 = galois.irreducible_polys(3, 4)
    g = galois.Poly.One(GF)
    for p in polys_1 + polys_2 + polys_4:
        g *= p
    x = galois.Poly.Identity(GF)
    assert g == x**(3**4) - x


def test_primitive_poly():
    p = galois.primitive_poly(2, 8)
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    assert galois.is_primitive(p)
    assert p == galois.conway_poly(2, 8)

    p = galois.primitive_poly(2, 8, method="max")
    assert galois.is_monic(p)
    assert galois.is_irreducible(p)
    assert galois.is_primitive(p)


def test_primitive_polys():
    """
    Generated with Octave:
        primpoly(5, 'all')
        primpoly(6, 'all')
        primpoly(7, 'all')
        primpoly(8, 'all')
    """
    assert [p.integer for p in galois.primitive_polys(2, 1)] == [3]
    assert [p.integer for p in galois.primitive_polys(2, 2)] == [7]
    assert [p.integer for p in galois.primitive_polys(2, 3)] == [11, 13]
    assert [p.integer for p in galois.primitive_polys(2, 4)] == [19, 25]
    assert [p.integer for p in galois.primitive_polys(2, 5)] == [37, 41, 47, 55, 59, 61]
    assert [p.integer for p in galois.primitive_polys(2, 6)] == [67, 91, 97, 103, 109, 115]
    assert [p.integer for p in galois.primitive_polys(2, 7)] == [131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213, 229, 239, 241, 247, 253]
    assert [p.integer for p in galois.primitive_polys(2, 8)] == [285, 299, 301, 333, 351, 355, 357, 361, 369, 391, 397, 425, 451, 463, 487, 501]

    # https://oeis.org/A011260
    assert len(galois.primitive_polys(2, 1)) == 1
    assert len(galois.primitive_polys(2, 2)) == 1
    assert len(galois.primitive_polys(2, 3)) == 2
    assert len(galois.primitive_polys(2, 4)) == 2
    assert len(galois.primitive_polys(2, 5)) == 6
    assert len(galois.primitive_polys(2, 6)) == 6
    assert len(galois.primitive_polys(2, 7)) == 18
    assert len(galois.primitive_polys(2, 8)) == 16
    assert len(galois.primitive_polys(2, 9)) == 48
    assert len(galois.primitive_polys(2, 10)) == 60

    # https://baylor-ir.tdl.org/bitstream/handle/2104/8793/GF3%20Polynomials.pdf?sequence=1&isAllowed=y
    GF = galois.GF(3)
    assert galois.primitive_polys(3, 2) == [
        galois.Poly([1,1,2], field=GF),
        galois.Poly([1,2,2], field=GF),
    ]
    assert galois.primitive_polys(3, 3) == [
        galois.Poly([1,0,2,1], field=GF),
        galois.Poly([1,1,2,1], field=GF),
        galois.Poly([1,2,0,1], field=GF),
        galois.Poly([1,2,1,1], field=GF),
    ]
    assert galois.primitive_polys(3, 4) == [
        galois.Poly([1,0,0,1,2], field=GF),
        galois.Poly([1,0,0,2,2], field=GF),
        galois.Poly([1,1,0,0,2], field=GF),
        galois.Poly([1,1,1,2,2], field=GF),
        galois.Poly([1,1,2,2,2], field=GF),
        galois.Poly([1,2,0,0,2], field=GF),
        galois.Poly([1,2,1,1,2], field=GF),
        galois.Poly([1,2,2,1,2], field=GF),
    ]
    assert galois.primitive_polys(3, 5) == [
        galois.Poly([1,0,0,0,2,1], field=GF),
        galois.Poly([1,0,0,2,1,1], field=GF),
        galois.Poly([1,0,1,0,1,1], field=GF),
        galois.Poly([1,0,1,2,0,1], field=GF),
        galois.Poly([1,0,1,2,2,1], field=GF),
        galois.Poly([1,0,2,1,0,1], field=GF),
        galois.Poly([1,0,2,2,1,1], field=GF),
        galois.Poly([1,1,0,0,2,1], field=GF),
        galois.Poly([1,1,0,1,0,1], field=GF),
        galois.Poly([1,1,0,1,1,1], field=GF),
        galois.Poly([1,1,1,0,1,1], field=GF),
        galois.Poly([1,1,1,1,2,1], field=GF),
        galois.Poly([1,1,1,2,1,1], field=GF),
        galois.Poly([1,1,2,0,0,1], field=GF),
        galois.Poly([1,1,2,1,1,1], field=GF),
        galois.Poly([1,1,2,2,0,1], field=GF),
        galois.Poly([1,2,0,0,0,1], field=GF),
        galois.Poly([1,2,0,0,1,1], field=GF),
        galois.Poly([1,2,0,2,2,1], field=GF),
        galois.Poly([1,2,1,1,1,1], field=GF),
        galois.Poly([1,2,2,0,2,1], field=GF),
        galois.Poly([1,2,2,1,0,1], field=GF),
    ]


def test_matlab_primitive_poly_GF2():
    """
    Generated with Matlab:
        gfprimdf(m, 2)
    """
    assert galois.matlab_primitive_poly(2, 1) == galois.Poly([1,1], order="asc")
    assert galois.matlab_primitive_poly(2, 2) == galois.Poly([1,1,1], order="asc")
    assert galois.matlab_primitive_poly(2, 3) == galois.Poly([1,1,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 4) == galois.Poly([1,1,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 5) == galois.Poly([1,0,1,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 6) == galois.Poly([1,1,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 7) == galois.Poly([1,0,0,1,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 8) == galois.Poly([1,0,1,1,1,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 9) == galois.Poly([1,0,0,0,1,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 10) == galois.Poly([1,0,0,1,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 11) == galois.Poly([1,0,1,0,0,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 12) == galois.Poly([1,1,0,0,1,0,1,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 13) == galois.Poly([1,1,0,1,1,0,0,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 14) == galois.Poly([1,1,0,0,0,0,1,0,0,0,1,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 15) == galois.Poly([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 16) == galois.Poly([1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1], order="asc")


def test_matlab_primitive_poly_GF3():
    """
    Generated with Matlab:
        gfprimdf(m, 3)
    """
    GF = galois.GF(3)
    assert galois.matlab_primitive_poly(3, 1) == galois.Poly([1,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(3, 2) == galois.Poly([2,1,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(3, 3) == galois.Poly([1,2,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(3, 4) == galois.Poly([2,1,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(3, 5) == galois.Poly([1,2,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(3, 6) == galois.Poly([2,1,0,0,0,0,1], field=GF, order="asc")
    # assert galois.matlab_primitive_poly(3, 7) == galois.Poly([1,0,2,0,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(3, 8) == galois.Poly([2,0,0,1,0,0,0,0,1], field=GF, order="asc")


def test_matlab_primitive_poly_GF5():
    """
    Generated with Matlab:
        gfprimdf(m, 5)
    """
    GF = galois.GF(5)
    assert galois.matlab_primitive_poly(5, 1) == galois.Poly([2,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(5, 2) == galois.Poly([2,1,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(5, 3) == galois.Poly([2,3,0,1], field=GF, order="asc")
    # assert galois.matlab_primitive_poly(5, 4) == galois.Poly([2,2,1,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(5, 5) == galois.Poly([2,4,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(5, 6) == galois.Poly([2,1,0,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(5, 7) == galois.Poly([2,3,0,0,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(5, 8) == galois.Poly([3,2,1,0,0,0,0,0,1], field=GF, order="asc")


def test_matlab_primitive_poly_GF7():
    """
    Generated with Matlab:
        gfprimdf(m, 7)
    """
    GF = galois.GF(7)
    assert galois.matlab_primitive_poly(7, 1) == galois.Poly([2,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(7, 2) == galois.Poly([3,1,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(7, 3) == galois.Poly([2,3,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(7, 4) == galois.Poly([5,3,1,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(7, 5) == galois.Poly([4,1,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(7, 6) == galois.Poly([5,1,3,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(7, 7) == galois.Poly([2,6,0,0,0,0,0,1], field=GF, order="asc")
    assert galois.matlab_primitive_poly(7, 8) == galois.Poly([3,1,0,0,0,0,0,0,1], field=GF, order="asc")


def test_minimal_poly():
    GF = galois.GF(5)
    alpha = GF.primitive_element
    p = galois.minimal_poly(alpha)
    assert p.field == GF
    assert p == galois.Poly([1, 3], field=GF)

    assert galois.minimal_poly(GF(0)) == galois.Poly.Identity(GF)
    assert galois.minimal_poly(GF(1)) == galois.Poly([1, 4], field=GF)
    assert galois.minimal_poly(GF(2)) == galois.Poly([1, 3], field=GF)
    assert galois.minimal_poly(GF(3)) == galois.Poly([1, 2], field=GF)

    GF = galois.GF(2**4)
    alpha = GF.primitive_element
    p = galois.minimal_poly(alpha)
    assert p.field == galois.GF2
    assert p == galois.Poly.Degrees([4, 1, 0])

    assert galois.minimal_poly(GF(0)) == galois.Poly.Identity()
    assert galois.minimal_poly(alpha**1) == galois.minimal_poly(alpha**2) == galois.minimal_poly(alpha**4) == galois.minimal_poly(alpha**8) == galois.Poly.Degrees([4, 1, 0])
    assert galois.minimal_poly(alpha**3) == galois.minimal_poly(alpha**6) == galois.minimal_poly(alpha**9) == galois.minimal_poly(alpha**12) == galois.Poly.Degrees([4, 3, 2, 1, 0])
    assert galois.minimal_poly(alpha**5) == galois.minimal_poly(alpha**10) == galois.Poly.Degrees([2, 1, 0])
    assert galois.minimal_poly(alpha**7) == galois.minimal_poly(alpha**11) == galois.minimal_poly(alpha**13) == galois.minimal_poly(alpha**14) == galois.Poly.Degrees([4, 3, 0])


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


def test_is_irreducible_GF2():
    p = galois.conway_poly(2, 4)
    assert galois.is_irreducible(p)

    p = galois.conway_poly(2, 5)
    assert galois.is_irreducible(p)

    # Has no roots in GF(2) but is still reducible
    p = galois.conway_poly(2, 4) * galois.conway_poly(2, 5)
    assert not galois.is_irreducible(p)


def test_is_irreducible_GF7():
    GF = galois.GF(7)

    p = galois.conway_poly(7, 4)
    assert galois.is_irreducible(p)
    assert galois.is_irreducible(p * GF.Random(low=2))

    p = galois.conway_poly(7, 5)
    assert galois.is_irreducible(p)
    assert galois.is_irreducible(p * GF.Random(low=2))

    # Has no roots in GF(7) but is still reducible
    p = galois.conway_poly(7, 4) * galois.conway_poly(7, 5)
    assert not galois.is_irreducible(p)

    p = galois.conway_poly(7, 20)
    assert galois.is_irreducible(p)
    assert galois.is_irreducible(p * GF.Random(low=2))

    p = galois.Poly.Roots([2, 3], field=GF)
    assert not galois.is_irreducible(p)

    x = galois.Poly.Identity(GF)
    p = galois.Poly.Random(20, field=GF) * x
    assert not galois.is_irreducible(p)

    # x + a is always irreducible over any Galois field
    p = galois.Poly([1, 0], field=GF)
    assert galois.is_irreducible(p)
    assert galois.is_irreducible(p * GF.Random(low=2))

    p = galois.Poly([1, GF.Random(low=1)], field=GF)
    assert galois.is_irreducible(p)
    assert galois.is_irreducible(p * GF.Random(low=2))


def test_is_irreducible_exceptions():
    with pytest.raises(TypeError):
        galois.is_irreducible(galois.GF2.Random(5))
    with pytest.raises(ValueError):
        galois.is_irreducible(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.is_irreducible(galois.Poly.Random(5, field=galois.GF(2**8)))


def test_is_primitive():
    assert galois.is_primitive(galois.conway_poly(2, 1))
    assert galois.is_primitive(galois.conway_poly(2, 2))
    assert galois.is_primitive(galois.conway_poly(2, 3))
    assert galois.is_primitive(galois.conway_poly(2, 4))
    assert galois.is_primitive(galois.conway_poly(2, 5))
    assert galois.is_primitive(galois.conway_poly(2, 6))
    assert galois.is_primitive(galois.conway_poly(2, 7))
    assert galois.is_primitive(galois.conway_poly(2, 8))

    # x^8 passes the primitivity tests but not irreducibility tests, needs to return False not True
    assert not galois.is_primitive(galois.Poly.Degrees([8]))

    # The AES irreducible polynomial is not primitive
    p = galois.Poly.Degrees([8,4,3,1,0])
    assert not galois.is_primitive(p)

    assert galois.is_primitive(galois.conway_poly(3, 1))
    assert galois.is_primitive(galois.conway_poly(3, 2))
    assert galois.is_primitive(galois.conway_poly(3, 3))
    assert galois.is_primitive(galois.conway_poly(3, 4))
    assert galois.is_primitive(galois.conway_poly(3, 5))

    # x + 1 is irreducible over GF(5) but not primitive
    GF = galois.GF(5)
    p = galois.Poly([1, 1], field=GF)
    assert not galois.is_primitive(p)

    # x + 2 is irreducible over GF(5) and primitive
    GF = galois.GF(5)
    p = galois.Poly([1, 2], field=GF)
    assert galois.is_primitive(p)


def test_is_primitive_exceptions():
    with pytest.raises(TypeError):
        galois.is_primitive(galois.GF2.Random(5))
    with pytest.raises(TypeError):
        galois.is_primitive(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.is_primitive(galois.Poly.Random(5, field=galois.GF(2**8)))


def test_primitive_element():
    x = galois.Poly.Identity()
    assert galois.primitive_element(galois.conway_poly(2, 2)) == x
    assert galois.primitive_element(galois.conway_poly(2, 3)) == x
    assert galois.primitive_element(galois.conway_poly(2, 4)) == x

    assert galois.primitive_element(galois.conway_poly(2, 2), reverse=True) == galois.Poly([1, 1])


def test_primitive_element_exceptions():
    p = galois.conway_poly(2, 8)

    with pytest.raises(TypeError):
        galois.primitive_element(p.coeffs)
    with pytest.raises(TypeError):
        galois.primitive_element(p, start=2.0)
    with pytest.raises(TypeError):
        galois.primitive_element(p, stop=256.0)
    with pytest.raises(TypeError):
        galois.primitive_element(p, reverse=1)
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Random(2)*galois.Poly.Random(2))
    with pytest.raises(ValueError):
        galois.primitive_element(p, start=200, stop=100)


def test_primitive_elements():
    assert galois.primitive_elements(galois.conway_poly(2, 2)) == [galois.Poly([1,0]), galois.Poly([1, 1])]


def test_primitive_elements_exceptions():
    p = galois.conway_poly(2, 8)

    with pytest.raises(TypeError):
        galois.primitive_elements(p.coeffs)
    with pytest.raises(TypeError):
        galois.primitive_elements(p, start=2.0)
    with pytest.raises(TypeError):
        galois.primitive_elements(p, stop=256.0)
    with pytest.raises(TypeError):
        galois.primitive_elements(p, reverse=1)
    with pytest.raises(ValueError):
        galois.primitive_elements(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.primitive_elements(galois.Poly.Random(2)*galois.Poly.Random(2))
    with pytest.raises(ValueError):
        galois.primitive_elements(p, start=200, stop=100)
