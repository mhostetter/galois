"""
A pytest module to test polynomial evaluation of matrices.

Sage:
    F = GF(3**5, name="x", repr="int")
    R = PolynomialRing(F, "x")
    X = []
    n = 2
    for i in range(n):
        row = []
        for j in range(n):
            row.append(F.random_element())
        X.append(row)
    print(X)
    X = matrix(X)
    for _ in range(3):
        p = R.random_element(5)
        print(p)
        print(list(p(X)))
"""
import pytest
import numpy as np

import galois


def test_exceptions():
    GF = galois.GF(2**8)
    p = galois.Poly([3,0,2,1], field=GF)
    x = GF([101, 102, 103, 104])

    with pytest.raises(TypeError):
        p(x, field="invalid type")
    with pytest.raises(ValueError):
        p(x, elementwise=False)


def test_binary_extension():
    GF = galois.GF(2**8)
    X = GF([[66, 232], [44, 46]])

    p = galois.Poly.String("19*x^5 + 10*x^4 + 145*x^3 + 143*x^2 + 133*x + 87", field=GF)
    assert np.array_equal(p(X, elementwise=False), GF([(178, 163), (190, 189)]))

    p = galois.Poly.String("216*x^5 + 48*x^4 + 250*x^3 + 181*x^2 + 182*x + 216", field=GF)
    assert np.array_equal(p(X, elementwise=False), GF([(243, 59), (113, 254)]))

    p = galois.Poly.String("121*x^5 + 165*x^4 + 184*x^3 + 198*x^2 + 248*x + 156", field=GF)
    assert np.array_equal(p(X, elementwise=False), GF([(24, 145), (66, 188)]))


def test_prime_extension():
    GF = galois.GF(3**5)
    X = GF([[170, 221], [175, 156]])

    p = galois.Poly.String("141*x^5 + 126*x^4 + 43*x^3 + 46*x^2 + 95*x + 106", field=GF)
    assert np.array_equal(p(X, elementwise=False), GF([(50, 179), (49, 133)]))

    p = galois.Poly.String("91*x^5 + 206*x^4 + 143*x^3 + 34*x^2 + 211*x + 162", field=GF)
    assert np.array_equal(p(X, elementwise=False), GF([(89, 156), (230, 139)]))

    p = galois.Poly.String("171*x^5 + 214*x^4 + 82*x^3 + x^2 + 97*x + 109", field=GF)
    assert np.array_equal(p(X, elementwise=False), GF([(170, 230), (202, 104)]))
