"""
A pytest module to test the quadratic (non-)residues in finite fields.

Sage:
    F = GF(79)
    qr = []
    qnr = []
    b = []
    for x in range(0, F.order()):
        x = F(x)
        sqrt_x = sqrt(x)
        if type(sqrt_x) == type(x):
            qr.append(x)
            b.append(True)
        else:
            qnr.append(x)
            b.append(False)
    print(qr)
    print(qnr)
    print(b)

Sage:
    F = GF(2**3, repr="int")
    qr = []
    qnr = []
    b = []
    for x in range(0, F.order()):
        x = F.fetch_int(x)
        try:
            sqrt_x = sqrt(x)
            qr.append(x)
            b.append(True)
        except:
            qnr.append(x)
            b.append(False)
    print(qr)
    print(qnr)
    print(b)
"""
import pytest
import numpy as np

import galois


def test_shapes():
    GF = galois.GF(31)
    x = GF.Random()
    assert isinstance(x.is_quadratic_residue(), np.bool_)
    x = GF.Random((2,))
    assert x.is_quadratic_residue().shape == x.shape
    x = GF.Random((2,2))
    assert x.is_quadratic_residue().shape == x.shape
    x = GF.Random((2,2,2))
    assert x.is_quadratic_residue().shape == x.shape

    GF = galois.GF(2**4)
    x = GF.Random()
    assert isinstance(x.is_quadratic_residue(), np.bool_)
    x = GF.Random((2,))
    assert x.is_quadratic_residue().shape == x.shape
    x = GF.Random((2,2))
    assert x.is_quadratic_residue().shape == x.shape
    x = GF.Random((2,2,2))
    assert x.is_quadratic_residue().shape == x.shape


def test_binary_field():
    GF = galois.GF(2)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True])
    assert isinstance(b, np.ndarray)


def test_prime_field_1():
    GF = galois.GF(7)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 4])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [3, 5, 6])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, False, True, False, False])
    assert isinstance(b, np.ndarray)


def test_prime_field_2():
    GF = galois.GF(31)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 4, 5, 7, 8, 9, 10, 14, 16, 18, 19, 20, 25, 28])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [3, 6, 11, 12, 13, 15, 17, 21, 22, 23, 24, 26, 27, 29, 30])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, False, True, True, False, True, True, True, True, False, False, False, True, False, True, False, True, True, True, False, False, False, False, True, False, False, True, False, False])
    assert isinstance(b, np.ndarray)


def test_prime_field_3():
    GF = galois.GF(79)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 4, 5, 8, 9, 10, 11, 13, 16, 18, 19, 20, 21, 22, 23, 25, 26, 31, 32, 36, 38, 40, 42, 44, 45, 46, 49, 50, 51, 52, 55, 62, 64, 65, 67, 72, 73, 76])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [3, 6, 7, 12, 14, 15, 17, 24, 27, 28, 29, 30, 33, 34, 35, 37, 39, 41, 43, 47, 48, 53, 54, 56, 57, 58, 59, 60, 61, 63, 66, 68, 69, 70, 71, 74, 75, 77, 78])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, False, True, True, False, False, True, True, True, True, False, True, False, False, True, False, True, True, True, True, True, True, False, True, True, False, False, False, False, True, True, False, False, False, True, False, True, False, True, False, True, False, True, True, True, False, False, True, True, True, True, False, False, True, False, False, False, False, False, False, True, False, True, True, False, True, False, False, False, False, True, True, False, False, True, False, False])
    assert isinstance(b, np.ndarray)


def test_binary_extension_field_1():
    GF = galois.GF(2**3)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 3, 4, 5, 6, 7])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, True, True, True, True, True])
    assert isinstance(b, np.ndarray)


def test_binary_extension_field_2():
    GF = galois.GF(2**4)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
    assert isinstance(b, np.ndarray)


def test_binary_extension_field_3():
    # Use an irreducible, but not primitive, polynomial
    GF = galois.GF(2**4, irreducible_poly="x^4 + x^3 + x^2 + x + 1")
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
    assert isinstance(b, np.ndarray)


def test_binary_extension_field_4():
    GF = galois.GF(2**5)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
    assert isinstance(b, np.ndarray)


def test_prime_extension_field_1():
    GF = galois.GF(3**2)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 4, 8])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [3, 5, 6, 7])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, False, True, False, False, False, True])
    assert isinstance(b, np.ndarray)


def test_prime_extension_field_2():
    GF = galois.GF(3**3)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 20, 22, 25])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [2, 3, 4, 5, 10, 14, 17, 18, 19, 21, 23, 24, 26])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, False, False, False, False, True, True, True, True, False, True, True, True, False, True, True, False, False, False, True, False, True, False, False, True, False])
    assert isinstance(b, np.ndarray)


def test_prime_extension_field_3():
    GF = galois.GF(5**2)
    x = GF.elements

    qr = GF.quadratic_residues
    assert np.array_equal(qr, [0, 1, 2, 3, 4, 6, 8, 11, 12, 18, 19, 22, 24])
    assert type(qr) is GF

    qnr = GF.quadratic_non_residues
    assert np.array_equal(qnr, [5, 7, 9, 10, 13, 14, 15, 16, 17, 20, 21, 23])
    assert type(qnr) is GF

    b = x.is_quadratic_residue()
    assert np.array_equal(b, [True, True, True, True, True, False, True, False, True, False, False, True, True, False, False, False, False, False, True, True, False, False, True, False, True])
    assert isinstance(b, np.ndarray)
