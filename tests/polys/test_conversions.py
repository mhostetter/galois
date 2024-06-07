"""
A pytest module to test the polynomial conversion functions.
"""

import random

import pytest

import galois


def test_integer_to_poly():
    assert galois._polys._conversions.integer_to_poly(0, 2) == [0]
    assert galois._polys._conversions.integer_to_poly(5, 2) == [1, 0, 1]
    assert galois._polys._conversions.integer_to_poly(5, 2, degree=3) == [0, 1, 0, 1]

    assert galois._polys._conversions.integer_to_poly(3**5, 3) == [1, 0, 0, 0, 0, 0]

    for _ in range(5):
        order = random.randint(2, 1_000_000_000)
        assert galois._polys._conversions.integer_to_poly(order**5 - 1, order) == [order - 1] * 5
        assert galois._polys._conversions.integer_to_poly(order**5, order) == [1, 0, 0, 0, 0, 0]
        assert galois._polys._conversions.integer_to_poly(order**5 + 1, order) == [1, 0, 0, 0, 0, 1]


def test_poly_to_integer():
    assert galois._polys._conversions.poly_to_integer([0], 2) == 0
    assert galois._polys._conversions.poly_to_integer([1, 0, 1], 2) == 5
    assert galois._polys._conversions.poly_to_integer([0, 1, 0, 1], 2) == 5

    assert galois._polys._conversions.poly_to_integer([1, 0, 0, 0, 0, 0], 3) == 3**5

    for _ in range(5):
        order = random.randint(2, 1_000_000_000)
        assert galois._polys._conversions.poly_to_integer([order - 1] * 5, order) == order**5 - 1
        assert galois._polys._conversions.poly_to_integer([1, 0, 0, 0, 0, 0], order) == order**5
        assert galois._polys._conversions.poly_to_integer([1, 0, 0, 0, 0, 1], order) == order**5 + 1


def test_sparse_poly_to_integer():
    assert galois._polys._conversions.sparse_poly_to_integer([0], [0], 2) == 0
    assert galois._polys._conversions.sparse_poly_to_integer([2, 0], [1, 1], 2) == 5


def test_poly_to_str():
    assert galois._polys._conversions.poly_to_str([0]) == "0"
    assert galois._polys._conversions.poly_to_str([1, 0, 1, 1]) == "x^3 + x + 1"
    assert galois._polys._conversions.poly_to_str([0, 1, 0, 1, 1]) == "x^3 + x + 1"

    assert galois._polys._conversions.poly_to_str([0], poly_var="y") == "0"
    assert galois._polys._conversions.poly_to_str([1, 0, 1, 1], poly_var="y") == "y^3 + y + 1"
    assert galois._polys._conversions.poly_to_str([0, 1, 0, 1, 1], poly_var="y") == "y^3 + y + 1"


def test_sparse_poly_to_str():
    assert galois._polys._conversions.sparse_poly_to_str([0], [0]) == "0"
    assert galois._polys._conversions.sparse_poly_to_str([3, 1, 0], [1, 1, 1]) == "x^3 + x + 1"

    assert galois._polys._conversions.sparse_poly_to_str([0], [0], poly_var="y") == "0"
    assert galois._polys._conversions.sparse_poly_to_str([3, 1, 0], [1, 1, 1], poly_var="y") == "y^3 + y + 1"

    GF = galois.GF(2**8)
    with GF.repr("poly"):
        assert galois._polys._conversions.sparse_poly_to_str([0], GF([0])) == "0"
        assert galois._polys._conversions.sparse_poly_to_str([3, 1, 0], GF([1, 2, 3])) == "x^3 + (α)x + (α + 1)"

        assert galois._polys._conversions.sparse_poly_to_str([0], GF([0]), poly_var="y") == "0"
        assert (
            galois._polys._conversions.sparse_poly_to_str([3, 1, 0], GF([1, 2, 3]), poly_var="y")
            == "y^3 + (α)y + (α + 1)"
        )


def test_str_to_sparse_poly():
    # Over GF(2)
    assert galois._polys._conversions.str_to_sparse_poly("x^2 + 1") == ([2, 0], [1, 1])
    assert galois._polys._conversions.str_to_sparse_poly("1 - x^2") == ([0, 2], [1, -1])
    assert galois._polys._conversions.str_to_sparse_poly("x**2 + 1") == ([2, 0], [1, 1])
    assert galois._polys._conversions.str_to_sparse_poly("y^2 + y + 1") == ([2, 1, 0], [1, 1, 1])
    assert galois._polys._conversions.str_to_sparse_poly("y**2 + y**1 + 1*y**0") == ([2, 1, 0], [1, 1, 1])

    # Over GF(3)
    assert galois._polys._conversions.str_to_sparse_poly("2*x^2 + 2") == ([2, 0], [2, 2])
    assert galois._polys._conversions.str_to_sparse_poly("2*x^2 - 1") == ([2, 0], [2, -1])

    # Over GF(2)
    with pytest.raises(ValueError):
        galois._polys._conversions.str_to_sparse_poly("x^2 + y + 1")
    with pytest.raises(ValueError):
        galois._polys._conversions.str_to_sparse_poly("x^2 + x^-1 + 1")


def test_str_to_integer():
    GF = galois.GF2
    assert galois._polys._conversions.str_to_integer("x^2 + 1", GF) == 5
    assert galois._polys._conversions.str_to_integer("x**2 + 1", GF) == 5
    assert galois._polys._conversions.str_to_integer("y^2 + y + 1", GF) == 7
    assert galois._polys._conversions.str_to_integer("y**2 + y**1 + 1*y**0", GF) == 7

    GF = galois.GF(3)
    assert galois._polys._conversions.str_to_integer("2*x^2 + 2", GF) == 2 * 3**2 + 2
    assert galois._polys._conversions.str_to_integer("2*x^2 - 1", GF) == 2 * 3**2 + 2

    GF = galois.GF2
    with pytest.raises(ValueError):
        galois._polys._conversions.str_to_integer("x^2 + y + 1", GF)
    with pytest.raises(ValueError):
        galois._polys._conversions.str_to_integer("x^2 + x^-1 + 1", GF)
