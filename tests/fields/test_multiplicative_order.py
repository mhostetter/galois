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
        GF(0).multiplicative_order()
    with pytest.raises(ArithmeticError):
        GF.Elements().multiplicative_order()


def test_shapes():
    # NOTE: 1-D arrays are tested in other tests
    GF = galois.GF(3**3)
    x = GF.Range(1, GF.order)
    y_truth = [1, 2, 26, 26, 26, 13, 13, 13, 13, 26, 13, 13, 13, 26, 13, 13, 26, 26, 26, 13, 26, 13, 26, 26, 13, 26]

    # Scalar
    y0 = x[0].multiplicative_order()
    assert np.array_equal(y0, y_truth[0])
    assert isinstance(y0, np.integer)

    # N-D
    y = x.reshape((2,13)).multiplicative_order()
    assert np.array_equal(y, np.array(y_truth).reshape(2,13))
    assert isinstance(y, np.ndarray)


def test_binary_field():
    GF = galois.GF(2)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1,]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_field_1():
    GF = galois.GF(7)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 3, 6, 3, 6, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_field_2():
    GF = galois.GF(31)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 5, 30, 5, 3, 6, 15, 5, 15, 15, 30, 30, 30, 15, 10, 5, 30, 15, 15, 15, 30, 30, 10, 30, 3, 6, 10, 15, 10, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_field_3():
    GF = galois.GF(79)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 39, 78, 39, 39, 78, 78, 13, 39, 13, 39, 26, 39, 26, 26, 39, 26, 13, 39, 39, 13, 13, 3, 6, 39, 39, 26, 78, 78, 78, 39, 39, 26, 78, 78, 39, 78, 13, 78, 39, 26, 39, 78, 39, 39, 13, 78, 78, 39, 39, 39, 13, 78, 78, 3, 6, 26, 26, 78, 78, 26, 13, 78, 13, 13, 78, 13, 78, 26, 78, 26, 39, 39, 78, 78, 39, 78, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_1():
    GF = galois.GF(2**3)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 7, 7, 7, 7, 7, 7]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_2():
    GF = galois.GF(2**4)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 15, 15, 15, 15, 3, 3, 5, 15, 5, 15, 5, 15, 15, 5]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_3():
    # Use an irreducible, but not primitive, polynomial
    GF = galois.GF(2**4, irreducible_poly="x^4 + x^3 + x^2 + x + 1")
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 5, 15, 5, 15, 15, 15, 5, 15, 15, 15, 3, 3, 15, 5]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_4():
    GF = galois.GF(2**5)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_field_1():
    GF = galois.GF(3**2)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 2, 8, 4, 8, 8, 8, 4]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_field_2():
    GF = galois.GF(3**3)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 2, 26, 26, 26, 13, 13, 13, 13, 26, 13, 13, 13, 26, 13, 13, 26, 26, 26, 13, 26, 13, 26, 26, 13, 26]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_field_3():
    GF = galois.GF(5**2)
    x = GF.Range(1, GF.order)
    y = x.multiplicative_order()
    y_truth = [1, 4, 4, 2, 24, 12, 8, 12, 24, 24, 3, 6, 24, 8, 24, 8, 24, 3, 6, 24, 24, 12, 8, 12]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_property_1():
    GF = galois.GF(2**8)
    x = GF.Random(10, low=1)
    order = x.multiplicative_order()
    assert np.all(x**order == 1)
