"""
A pytest module to test the traces over finite fields.

Sage:
    F = GF(2**5, repr="int")
    y = []
    for x in range(0, F.order()):
        x = F.fetch_int(x)
        y.append(x.trace())
    print(y)
"""
import pytest
import numpy as np

import galois


def test_exceptions():
    with pytest.raises(TypeError):
        GF = galois.GF(3)
        x = GF.Elements()
        y = x.field_trace()


def test_shapes():
    # NOTE: 1-D arrays are tested in other tests
    GF = galois.GF(3**3)
    x = GF.Elements()
    y_truth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Scalar
    y0 = x[0].field_trace()
    assert np.array_equal(y0, y_truth[0])
    assert isinstance(y0, GF.prime_subfield)

    # N-D
    y = x.reshape((3,3,3)).field_trace()
    assert np.array_equal(y, np.array(y_truth).reshape(3,3,3))
    assert isinstance(y0, GF.prime_subfield)


def test_binary_extension_1():
    GF = galois.GF(2**3)
    x = GF.Elements()
    y = x.field_trace()
    y_truth = [0, 1, 0, 1, 0, 1, 0, 1]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, GF.prime_subfield)


def test_binary_extension_2():
    GF = galois.GF(2**4)
    x = GF.Elements()
    y = x.field_trace()
    y_truth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, GF.prime_subfield)


def test_binary_extension_3():
    # Use an irreducible, but not primitive, polynomial
    GF = galois.GF(2**4, irreducible_poly="x^4 + x^3 + x^2 + x + 1")
    x = GF.Elements()
    y = x.field_trace()
    y_truth = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, GF.prime_subfield)


def test_binary_extension_4():
    GF = galois.GF(2**5)
    x = GF.Elements()
    y = x.field_trace()
    y_truth = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, GF.prime_subfield)


def test_prime_extension_1():
    GF = galois.GF(3**2)
    x = GF.Elements()
    y = x.field_trace()
    y_truth = [0, 2, 1, 1, 0, 2, 2, 1, 0]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, GF.prime_subfield)


def test_prime_extension_2():
    GF = galois.GF(3**3)
    x = GF.Elements()
    y = x.field_trace()
    y_truth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, GF.prime_subfield)


def test_prime_extension_3():
    GF = galois.GF(5**2)
    x = GF.Elements()
    y = x.field_trace()
    y_truth = [0, 2, 4, 1, 3, 1, 3, 0, 2, 4, 2, 4, 1, 3, 0, 3, 0, 2, 4, 1, 4, 1, 3, 0, 2]
    assert np.array_equal(y, y_truth)
    assert isinstance(y, GF.prime_subfield)


def test_property_1():
    """
    L = GF(q^n)
    K = GF(q)
    a in L
    Tr(a^q) = Tr(a)
    """
    q = 7
    n = 3
    L = galois.GF(q**n)
    a = L.Random(10)
    assert np.array_equal((a**q).field_trace(), a.field_trace())
