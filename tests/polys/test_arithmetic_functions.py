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


def test_modular_power_exceptions():
    GF = galois.GF(7)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 20

    with pytest.raises(TypeError):
        galois.pow(f.coeffs, power, g)
    with pytest.raises(TypeError):
        galois.pow(f, float(power), g)
    with pytest.raises(TypeError):
        galois.pow(f, power, g.coeffs)
    with pytest.raises(ValueError):
        galois.pow(f, -power, g)
    with pytest.raises(ValueError):
        galois.pow(f, -power, galois.Poly(g.coeffs, field=galois.GF(31)))


def test_modular_power(poly_modular_power):
    GF, X, E, M, Z = poly_modular_power["GF"], poly_modular_power["X"], poly_modular_power["E"], poly_modular_power["M"], poly_modular_power["Z"]
    for i in range(len(X)):
        x = X[i]
        e = E[i]
        m = M[i]
        z = galois.pow(x, e, m)

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
