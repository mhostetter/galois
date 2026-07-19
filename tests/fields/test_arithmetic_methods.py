"""
A pytest module to test the accuracy of Galois field arithmetic methods/operations.
"""

import random

import numpy as np
import pytest

import galois


def test_additive_order(field_additive_order):
    GF, X, Z = field_additive_order["GF"], field_additive_order["X"], field_additive_order["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    z = x.additive_order()
    assert np.array_equal(z, Z)
    assert type(z) is np.ndarray


def test_multiplicative_order(field_multiplicative_order):
    GF, X, Z = field_multiplicative_order["GF"], field_multiplicative_order["X"], field_multiplicative_order["Z"]
    if GF.dtypes[-1] == np.object_:
        # FIXME: Skipping large fields because they're too slow
        return

    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    z = x.multiplicative_order()
    assert np.array_equal(z, Z)
    assert type(z) is np.ndarray

    with pytest.raises(ArithmeticError):
        GF(0).multiplicative_order()
    with pytest.raises(ArithmeticError):
        GF.Range(0, 2).multiplicative_order()


def test_issue_532():
    """
    https://github.com/mhostetter/galois/issues/532
    """
    GF = galois.GF(2**64 - 59)
    assert GF(1).multiplicative_order() == 1


def test_characteristic_poly_element(field_characteristic_poly_element):
    GF, X, Z = (
        field_characteristic_poly_element["GF"],
        field_characteristic_poly_element["X"],
        field_characteristic_poly_element["Z"],
    )
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.characteristic_poly()
        assert z == z_truth

    # Only 0-D arrays are allowed
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.characteristic_poly()


def test_characteristic_poly_matrix(field_characteristic_poly_matrix):
    GF, X, Z = (
        field_characteristic_poly_matrix["GF"],
        field_characteristic_poly_matrix["X"],
        field_characteristic_poly_matrix["Z"],
    )
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.characteristic_poly()
        assert z == z_truth

    # Only 2-D square arrays are allowed
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.characteristic_poly()
    with pytest.raises(ValueError):
        A = GF.Random((2, 3))
        A.characteristic_poly()


def test_minimal_poly_element(field_minimal_poly_element):
    GF, X, Z = field_minimal_poly_element["GF"], field_minimal_poly_element["X"], field_minimal_poly_element["Z"]
    for x, z_truth in zip(X, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        z = x.minimal_poly()
        assert z == z_truth

    # Only 0-D arrays are allowed
    with pytest.raises(ValueError):
        A = GF.Random(5)
        A.minimal_poly()


def test_characteristic_minimal_poly_1x1():
    # A 1x1 matrix [[a]] has characteristic and minimal polynomial x - a. Exercises the
    # cofactor-expansion base case that previously raised IndexError for 1x1 matrices.
    fields = [galois.GF(2), galois.GF(5), galois.GF(7), galois.GF(2**4), galois.GF(3**3)]
    for GF in fields:
        x = galois.Poly.Identity(GF)
        for a in GF.elements:
            A = GF([[a]])
            c_poly = A.characteristic_poly()
            m_poly = A.minimal_poly()
            # det(x*I - [[a]]) = x - a for both the characteristic and minimal polynomial
            assert c_poly == x - a
            assert m_poly == x - a
            assert c_poly.degree == 1 and c_poly.coeffs[0] == GF(1)  # monic, degree 1
            # Cayley-Hamilton: the matrix is a root of its own characteristic polynomial
            assert np.array_equal(c_poly(A, elementwise=False), GF.Zeros((1, 1)))

    # The added base case leaves larger matrices unchanged: c(x) = x^2 - tr(A) x + det(A).
    GF = galois.GF(7)
    A = GF([[2, 3], [1, 4]])
    assert A.characteristic_poly() == galois.Poly([1, -np.trace(A), np.linalg.det(A)], field=GF)


# def test_minimal_poly_matrix(field_minimal_poly_matrix):
#     GF, X, Z = field_minimal_poly_matrix["GF"], field_minimal_poly_matrix["X"], field_minimal_poly_matrix["Z"]

#     for i in range(len(X)):
#         dtype = random.choice(GF.dtypes)
#         xi = X[i].astype(dtype)
#         zi = xi.minimal_poly()
#         assert zi == Z[i]

#     # Only 2-D square arrays are allowed
#     with pytest.raises(ValueError):
#         A = GF.Random(5)
#         A.minimal_poly()
#     with pytest.raises(ValueError):
#         A = GF.Random((2,3))
#         A.minimal_poly()


def test_field_trace(field_trace):
    GF, X, Z = field_trace["GF"], field_trace["X"], field_trace["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    z = x.field_trace()
    assert np.array_equal(z, Z)
    assert type(z) is GF.prime_subfield


def test_field_norm(field_norm):
    GF, X, Z = field_norm["GF"], field_norm["X"], field_norm["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    z = x.field_norm()
    assert np.array_equal(z, Z)
    assert type(z) is GF.prime_subfield
