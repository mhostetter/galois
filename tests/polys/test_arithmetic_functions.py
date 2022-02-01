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
