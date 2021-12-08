"""
A pytest module to test the multiplicative order of elements of finite fields.

Sage:
    F = GF(2**3)
    y = []
    for x in range(1, F.order()):
        x = F.fetch_int(x)
        y.append(x.multiplicative_order())
    print(y)
"""
import pytest
import numpy as np

import galois


def test_exceptions():
    GF = galois.GF(3)
    with pytest.raises(ArithmeticError):
        GF(0).order()
    with pytest.raises(ArithmeticError):
        GF.Elements().order()


def test_shapes():
    # NOTE: 1-D arrays are tested in other tests
    GF = galois.GF(3**3)
    x = GF.Range(1, GF.order)
    y_truth = [1, 2, 26, 26, 26, 13, 13, 13, 13, 26, 13, 13, 13, 26, 13, 13, 26, 26, 26, 13, 26, 13, 26, 26, 13, 26]

    # Scalar
    y0 = x[0].order()
    assert np.array_equal(y0, y_truth[0])
    assert isinstance(y0, np.integer)

    # N-D
    y = x.reshape((2,13)).order()
    assert np.array_equal(y, np.array(y_truth).reshape(2,13))
    assert isinstance(y, np.ndarray)


def test_binary_extension_1():
    GF = galois.GF(2**3)
    x = GF.Range(1, GF.order)
    y = x.order()
    y_truth = [1, 7, 7, 7, 7, 7, 7]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_2():
    GF = galois.GF(2**4)
    x = GF.Range(1, GF.order)
    y = x.order()
    y_truth = [1, 15, 15, 15, 15, 3, 3, 5, 15, 5, 15, 5, 15, 15, 5]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_3():
    # Use an irreducible, but not primitive, polynomial
    GF = galois.GF(2**4, irreducible_poly="x^4 + x^3 + x^2 + x + 1")
    x = GF.Range(1, GF.order)
    y = x.order()
    y_truth = [1, 5, 15, 5, 15, 15, 15, 5, 15, 15, 15, 3, 3, 15, 5]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_4():
    GF = galois.GF(2**5)
    x = GF.Range(1, GF.order)
    y = x.order()
    y_truth = [1, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_1():
    GF = galois.GF(3**2)
    x = GF.Range(1, GF.order)
    y = x.order()
    y_truth = [1, 2, 8, 4, 8, 8, 8, 4]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_2():
    GF = galois.GF(3**3)
    x = GF.Range(1, GF.order)
    y = x.order()
    y_truth = [1, 2, 26, 26, 26, 13, 13, 13, 13, 26, 13, 13, 13, 26, 13, 13, 26, 26, 26, 13, 26, 13, 26, 26, 13, 26]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_3():
    GF = galois.GF(5**2)
    x = GF.Range(1, GF.order)
    y = x.order()
    y_truth = [1, 4, 4, 2, 24, 12, 8, 12, 24, 24, 3, 6, 24, 8, 24, 8, 24, 3, 6, 24, 24, 12, 8, 12]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_property_1():
    GF = galois.GF(2**8)
    x = GF.Random(10, low=1)
    order = x.order()
    assert np.all(x**order == 1)
