"""
A pytest module to test the additive order of elements of finite fields.

Sage:
    F = GF(2**3)
    y = []
    for x in range(0, F.order()):
        x = F.fetch_int(x)
        y.append(x.additive_order())
    print(y)
"""
import pytest
import numpy as np

import galois


# def test_exceptions():
#     # None currently


def test_shapes():
    # NOTE: 1-D arrays are tested in other tests
    GF = galois.GF(3**3)
    x = GF.Elements()
    y_truth = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    # Scalar
    y0 = x[0].additive_order()
    assert np.array_equal(y0, y_truth[0])
    assert isinstance(y0, np.integer)

    # N-D
    y = x.reshape((3,3,3)).additive_order()
    assert np.array_equal(y, np.array(y_truth).reshape(3,3,3))
    assert isinstance(y, np.ndarray)


def test_binary_field():
    GF = galois.GF(2)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_field_1():
    GF = galois.GF(7)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 7, 7, 7, 7, 7, 7]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_field_2():
    GF = galois.GF(31)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_field_3():
    GF = galois.GF(79)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_1():
    GF = galois.GF(2**3)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 2, 2, 2, 2, 2, 2, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_2():
    GF = galois.GF(2**4)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_3():
    # Use an irreducible, but not primitive, polynomial
    GF = galois.GF(2**4, irreducible_poly="x^4 + x^3 + x^2 + x + 1")
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_binary_extension_field_4():
    GF = galois.GF(2**5)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_field_1():
    GF = galois.GF(3**2)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 3, 3, 3, 3, 3, 3, 3, 3]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_field_2():
    GF = galois.GF(3**3)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_prime_extension_field_3():
    GF = galois.GF(5**2)
    x = GF.Elements()
    y = x.additive_order()
    y_truth = [1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, np.ndarray)


def test_property_1():
    GF = galois.GF(2**8)
    x = GF.Random(10)
    order = x.additive_order()
    assert np.all(x*order == 0)
