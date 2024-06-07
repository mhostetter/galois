"""
A pytest module to test polynomial arithmetic functions.
"""

import numpy as np
import pytest

import galois


def test_gcd_exceptions():
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])

    with pytest.raises(TypeError):
        galois.gcd(a.coeffs, b)
    with pytest.raises(TypeError):
        galois.gcd(a, b.coeffs)
    with pytest.raises(ValueError):
        galois.gcd(a, galois.Poly(b.coeffs, field=galois.GF(3)))


def test_gcd(poly_egcd):
    GF, X, Y, D = poly_egcd["GF"], poly_egcd["X"], poly_egcd["Y"], poly_egcd["D"]
    for x, y, d_truth in zip(X, Y, D):
        d = galois.gcd(x, y)
        assert d == d_truth
        assert isinstance(d, galois.Poly)
        assert d.field is GF
        assert type(d.coeffs) is GF


def test_egcd_exceptions():
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])

    with pytest.raises(TypeError):
        galois.egcd(a.coeffs, b)
    with pytest.raises(TypeError):
        galois.egcd(a, b.coeffs)
    with pytest.raises(ValueError):
        galois.egcd(a, galois.Poly(b.coeffs, field=galois.GF(3)))


def test_egcd(poly_egcd):
    GF, X, Y, D, S, T = poly_egcd["GF"], poly_egcd["X"], poly_egcd["Y"], poly_egcd["D"], poly_egcd["S"], poly_egcd["T"]
    for x, y, d_truth, s_truth, t_truth in zip(X, Y, D, S, T):
        d, s, t = galois.egcd(x, y)

        assert d == d_truth
        assert isinstance(d, galois.Poly)
        assert d.field is GF
        assert type(d.coeffs) is GF

        assert s == s_truth
        assert isinstance(s, galois.Poly)
        assert s.field is GF
        assert type(s.coeffs) is GF

        assert t == t_truth
        assert isinstance(t, galois.Poly)
        assert t.field is GF
        assert type(t.coeffs) is GF


def test_lcm_exceptions():
    with pytest.raises(ValueError):
        a = galois.Poly.Random(5)
        b = galois.Poly.Random(4, field=galois.GF(3))
        c = galois.Poly.Random(3)
        galois.lcm(a, b, c)


def test_lcm(poly_lcm):
    GF, X, Z = poly_lcm["GF"], poly_lcm["X"], poly_lcm["Z"]
    for x, z_truth in zip(X, Z):
        z = galois.lcm(*x)
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_prod_exceptions():
    with pytest.raises(ValueError):
        a = galois.Poly.Random(5)
        b = galois.Poly.Random(4, field=galois.GF(3))
        c = galois.Poly.Random(3)
        galois.prod(a, b, c)


def test_prod(poly_prod):
    GF, X, Z = poly_prod["GF"], poly_prod["X"], poly_prod["Z"]
    for x, z_truth in zip(X, Z):
        z = galois.prod(*x)
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_crt_exceptions():
    a = [galois.Poly([1, 1]), galois.Poly([1, 1, 1])]
    m = [galois.Poly([1, 0, 1]), galois.Poly([1, 1, 0, 1])]

    with pytest.raises(TypeError):
        galois.crt(np.array(a, dtype=object), m)
    with pytest.raises(TypeError):
        galois.crt(a, np.array(m, dtype=object))
    with pytest.raises(TypeError):
        aa = a.copy()
        aa[0] = a[0].coeffs
        galois.crt(aa, m)
    with pytest.raises(TypeError):
        mm = m.copy()
        mm[0] = m[0].coeffs
        galois.crt(a, mm)
    with pytest.raises(ValueError):
        aa = a.copy()
        aa.append(galois.Poly([1, 1, 1]))
        galois.crt(aa, m)
    with pytest.raises(ValueError):
        mm = m.copy()
        mm.append(galois.Poly([1, 1, 0, 1]))
        galois.crt(a, mm)


def test_crt(poly_crt):
    GF, X, Y, Z = poly_crt["GF"], poly_crt["X"], poly_crt["Y"], poly_crt["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        if z_truth is not None:
            z = galois.crt(x, y)
            assert z == z_truth
            assert isinstance(z, galois.Poly)
            assert z.field is GF
            assert type(z.coeffs) is GF
        else:
            with pytest.raises(ValueError):
                galois.crt(x, y)
