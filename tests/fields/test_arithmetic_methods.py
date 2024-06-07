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
