"""
A pytest module to test generating Lagrange polynomials.

Sage:
    F = GF(2**8, repr="int")
    points = [(F.random_element(), F.random_element()) for _ in range(10)]
    print(list(zip(*points)))
    R = F['x']
    R.lagrange_polynomial(points)
"""
import pytest
import numpy as np

import galois


def test_exceptions():
    GF = galois.GF(251)
    x = GF([0, 1, 2, 3])
    y = GF([100, 101, 102, 103])

    with pytest.raises(TypeError):
        galois.lagrange_poly(x.view(np.ndarray), y)
    with pytest.raises(TypeError):
        galois.lagrange_poly(x, y.view(np.ndarray))
    with pytest.raises(TypeError):
        GF_other = galois.GF(2**8)
        galois.lagrange_poly(x, GF_other(y))
    with pytest.raises(ValueError):
        galois.lagrange_poly(x.reshape((2,2)), y)
    with pytest.raises(ValueError):
        galois.lagrange_poly(x, y.reshape((2,2)))
    with pytest.raises(ValueError):
        galois.lagrange_poly(x, np.append(y, 104))
    with pytest.raises(ValueError):
        galois.lagrange_poly(GF([0, 1, 2, 0]), y)


def test_lagrange_poly(poly_lagrange_poly):
    GF, X, Y, Z = poly_lagrange_poly["GF"], poly_lagrange_poly["X"], poly_lagrange_poly["Y"], poly_lagrange_poly["Z"]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        poly = galois.lagrange_poly(x, y)
        assert type(poly) == galois.Poly
        assert poly.field == GF
        assert poly == Z[i]
