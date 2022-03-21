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


def test_binary_field():
    GF = galois.GF(2)
    x = GF([0, 1])
    y = GF([0, 0])
    assert galois.lagrange_poly(x, y) == galois.Poly.Str("0", field=GF)

    GF = galois.GF(2)
    x = GF([0, 1])
    y = GF([0, 1])
    assert galois.lagrange_poly(x, y) == galois.Poly.Str("x", field=GF)

    GF = galois.GF(2)
    x = GF([0, 1])
    y = GF([1, 0])
    assert galois.lagrange_poly(x, y) == galois.Poly.Str("x + 1", field=GF)

    x = GF([0, 1])
    y = GF([1, 1])
    assert galois.lagrange_poly(x, y) == galois.Poly.Str("1", field=GF)


def test_prime_field():
    GF = galois.GF(251)
    x = GF([10, 117, 142, 120, 163, 13, 37, 67, 135, 55])
    y = GF([197, 191, 31, 24, 65, 222, 224, 58, 47, 178])
    assert galois.lagrange_poly(x, y) == galois.Poly.Str("207*x^9 + 222*x^8 + 227*x^7 + 88*x^6 + 6*x^5 + 52*x^4 + 82*x^3 + 198*x^2 + 27*x + 78", field=GF)


def test_binary_extension_field():
    GF = galois.GF(2**8)
    x = GF([141, 144, 248, 59, 208, 32, 254, 46, 230, 15])
    y = GF([250, 27, 81, 177, 62, 208, 221, 76, 182, 6])
    assert galois.lagrange_poly(x, y) == galois.Poly.Str("104*x^9 + 130*x^8 + 92*x^7 + 199*x^6 + 64*x^5 + 211*x^4 + 130*x^3 + 38*x^2 + 140*x + 114", field=GF)


def test_prime_extension_field():
    GF = galois.GF(3**5)
    x = GF([114, 151, 235, 198, 129, 192, 73, 184, 186, 78])
    y = GF([152, 50, 232, 129, 212, 226, 152, 26, 148, 239])
    assert galois.lagrange_poly(x, y) == galois.Poly.Str("42*x^9 + 200*x^8 + 109*x^7 + 82*x^6 + x^5 + 241*x^4 + 65*x^3 + 153*x^2 + 143*x + 88", field=GF)
