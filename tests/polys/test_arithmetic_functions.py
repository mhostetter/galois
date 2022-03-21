"""
A pytest module to test polynomial arithmetic functions.
"""
import pytest
import numpy as np

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
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        d = galois.gcd(x, y)

        assert d == D[i]
        assert isinstance(d, galois.Poly)


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
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        d, s, t = galois.egcd(x, y)

        assert d == D[i]
        assert isinstance(d, galois.Poly)
        assert s == S[i]
        assert isinstance(s, galois.Poly)
        assert t == T[i]
        assert isinstance(t, galois.Poly)


def test_lcm_exceptions():
    with pytest.raises(ValueError):
        a = galois.Poly.Random(5)
        b = galois.Poly.Random(4, field=galois.GF(3))
        c = galois.Poly.Random(3)
        galois.lcm(a, b, c)


def test_lcm(poly_lcm):
    GF, X, Z = poly_lcm["GF"], poly_lcm["X"], poly_lcm["Z"]
    for i in range(len(X)):
        x = X[i]
        z = galois.lcm(*x)

        assert z == Z[i]
        assert isinstance(z, galois.Poly)


def test_prod_exceptions():
    with pytest.raises(ValueError):
        a = galois.Poly.Random(5)
        b = galois.Poly.Random(4, field=galois.GF(3))
        c = galois.Poly.Random(3)
        galois.prod(a, b, c)


def test_prod(poly_prod):
    GF, X, Z = poly_prod["GF"], poly_prod["X"], poly_prod["Z"]
    for i in range(len(X)):
        x = X[i]
        z = galois.prod(*x)

        assert z == Z[i]
        assert isinstance(z, galois.Poly)


def test_crt_exceptions():
    a = [galois.Poly([1,1]), galois.Poly([1,1,1])]
    m = [galois.Poly([1,0,1]), galois.Poly([1,1,0,1])]

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
        aa.append(galois.Poly([1,1,1]))
        galois.crt(aa, m)
    with pytest.raises(ValueError):
        mm = m.copy()
        mm.append(galois.Poly([1,1,0,1]))
        galois.crt(a, mm)


def test_crt(poly_crt):
    X, Y, Z = poly_crt["X"], poly_crt["Y"], poly_crt["Z"]
    for i in range(len(X)):
        if Z[i] is not None:
            assert galois.crt(X[i], Y[i]) == Z[i]
        else:
            with pytest.raises(ValueError):
                galois.crt(X[i], Y[i])
