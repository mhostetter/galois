"""
A pytest module to test the characteristic polynomial of a square matrix.

Sage:
    F = GF(3**5, repr="int")
    A = []
    n = 2
    for i in range(n):
        row = []
        for j in range(n):
            row.append(F.fetch_int(randint(1, F.order())))
        A.append(row)
    print(A)
    A = matrix(A)
    print(A.charpoly())
"""
import pytest
import numpy as np

import galois


def test_exceptions():
    GF = galois.GF(2**8)
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.characteristic_poly()
    with pytest.raises(ValueError):
        A = GF.Random((4,5))
        A.characteristic_poly()


def test_2x2():
    GF = galois.GF(2**8)
    A = GF([[242, 238], [10, 228]])
    poly = galois.Poly.String("x^2 + 22*x + 220", field=GF)
    assert A.characteristic_poly() == poly

    GF = galois.GF(3**5)
    A = GF([[188, 184], [78, 39]])
    poly = galois.Poly.String("x^2 + 136*x + 139", field=GF)
    assert A.characteristic_poly() == poly


def test_3x3():
    GF = galois.GF(2**8)
    A = GF([[227, 160, 206], [153, 242, 208], [41, 153, 83]])
    poly = galois.Poly.String("x^3 + 66*x^2 + 254*x + 16", field=GF)
    assert A.characteristic_poly() == poly

    GF = galois.GF(3**5)
    A = GF([[208, 235, 208], [60, 132, 142], [91, 79, 193]])
    poly = galois.Poly.String("x^3 + 100*x^2 + 100*x + 164", field=GF)
    assert A.characteristic_poly() == poly


def test_4x4():
    GF = galois.GF(2**8)
    A = GF([[98, 210, 148, 132], [243, 227, 215, 187], [88, 241, 186, 225], [191, 109, 185, 39]])
    poly = galois.Poly.String("x^4 + 28*x^3 + 159*x^2 + 184*x + 43", field=GF)
    assert A.characteristic_poly() == poly

    GF = galois.GF(3**5)
    A = GF([[19, 103, 61, 183], [235, 126, 119, 150], [226, 90, 169, 30], [185, 187, 54, 215]])
    poly = galois.Poly.String("x^4 + 116*x^3 + 208*x^2 + 19*x + 133", field=GF)
    assert A.characteristic_poly() == poly


def test_5x5():
    GF = galois.GF(2**8)
    A = GF([[161, 250, 106, 129, 28], [227, 186, 241, 38, 229], [87, 243, 246, 252, 238], [26, 221, 188, 183, 30], [190, 19, 97, 110, 202]])
    poly = galois.Poly.String("x^5 + 144*x^4 + 197*x^3 + 123*x^2 + 122*x + 153", field=GF)
    assert A.characteristic_poly() == poly

    GF = galois.GF(3**5)
    A = GF([[205, 151, 28, 234, 237], [187, 205, 23, 201, 177], [133, 167, 56, 192, 168], [44, 167, 127, 15, 126], [34, 11, 88, 86, 15]])
    poly = galois.Poly.String("x^5 + 239*x^4 + 3*x^3 + 29*x^2 + 118*x + 103", field=GF)
    assert A.characteristic_poly() == poly


def test_6x6():
    GF = galois.GF(2**8)
    A = GF([[106, 204, 100, 114, 255, 156], [177, 52, 157, 96, 41, 49], [140, 183, 80, 216, 194, 75], [52, 241, 143, 47, 91, 112], [50, 56, 135, 120, 177, 128], [3, 115, 180, 212, 216, 137]])
    poly = galois.Poly.String("x^6 + 25*x^5 + 62*x^4 + 168*x^3 + 79*x^2 + 9*x + 152", field=GF)
    assert A.characteristic_poly() == poly

    GF = galois.GF(3**5)
    A = GF([[200, 6, 74, 9, 150, 111], [41, 36, 116, 208, 206, 80], [168, 180, 9, 112, 188, 116], [65, 170, 113, 61, 227, 210], [194, 68, 154, 240, 145, 9], [167, 55, 64, 23, 55, 18]])
    poly = galois.Poly.String("x^6 + 5*x^5 + 200*x^4 + 205*x^3 + 210*x^2 + 157*x + 107", field=GF)
    assert A.characteristic_poly() == poly
