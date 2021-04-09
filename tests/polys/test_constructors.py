"""
A pytest module to test Galois field polynomial alternate constructors.
"""
import numpy as np
import pytest

import galois


POLY_CLASSES = [galois.Poly, galois.poly.DensePoly, galois.poly.SparsePoly]

FIELDS = [
    galois.GF2,  # GF(2)
    galois.GF(31),  # GF(p) with np.int dtypes
    galois.GF(36893488147419103183),  # GF(p) with object dtype
    galois.GF(2**8),  # GF(2^m) with np.int dtypes
    galois.GF(2**100),  # GF(2^m) with object dtype
]


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_zero(poly, field):
    p = poly.Zero(field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 0
    assert np.array_equal(p.nonzero_degrees, [])
    assert np.array_equal(p.nonzero_coeffs, [])
    assert np.array_equal(p.degrees, [0])
    assert np.array_equal(p.coeffs, [0])
    assert p.integer == 0


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_one(poly, field):
    p = poly.One(field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 0
    assert np.array_equal(p.nonzero_degrees, [0])
    assert np.array_equal(p.nonzero_coeffs, [1])
    assert np.array_equal(p.degrees, [0])
    assert np.array_equal(p.coeffs, [1])
    assert p.integer == 1


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_identity(poly, field):
    p = poly.Identity(field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 1
    assert np.array_equal(p.nonzero_degrees, [1])
    assert np.array_equal(p.nonzero_coeffs, [1])
    assert np.array_equal(p.degrees, [1,0])
    assert np.array_equal(p.coeffs, [1,0])
    assert p.integer == field.order


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_random(poly, field):
    p = poly.Random(2, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_integer(poly, field):
    integer = field.order + 1  # Corresponds to p(x) = x + 1
    p = poly.Integer(integer, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 1
    assert np.array_equal(p.nonzero_degrees, [1,0])
    assert np.array_equal(p.nonzero_coeffs, [1,1])
    assert np.array_equal(p.degrees, [1,0])
    assert np.array_equal(p.coeffs, [1,1])
    assert p.integer == integer


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_degrees(poly, field):
    # Corresponds to p(x) = x^2 + 1
    degrees = [2,0]
    coeffs = [1,1]
    p = poly.Degrees(degrees, coeffs, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2
    assert np.array_equal(p.nonzero_degrees, [2,0])
    assert np.array_equal(p.nonzero_coeffs, [1,1])
    assert np.array_equal(p.degrees, [2,1,0])
    assert np.array_equal(p.coeffs, [1,0,1])
    assert p.integer == field.order**2 + 1


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_coeffs(poly, field):
    coeffs = [1,0,1]  # Corresponds to p(x) = x^2 + 1
    p = poly.Coeffs(coeffs, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2
    assert np.array_equal(p.nonzero_degrees, [2,0])
    assert np.array_equal(p.nonzero_coeffs, [1,1])
    assert np.array_equal(p.degrees, [2,1,0])
    assert np.array_equal(p.coeffs, [1,0,1])
    assert p.integer == field.order**2 + 1


@pytest.mark.parametrize("poly", POLY_CLASSES)
@pytest.mark.parametrize("field", FIELDS)
def test_roots(poly, field):
    a, b = field.Random(), field.Random()
    roots = [a, b]  # p(x) = (x - a)*(x - b)
    degree = 2
    degrees = [2, 1, 0]
    coeffs = [1, -a + -b, (-a)*(-b)]
    nonzero_degrees = [d for d, c in zip(degrees, coeffs) if c > 0]
    nonzero_coeffs = [c for d, c in zip(degrees, coeffs) if c > 0]
    integer = sum([int(c)*field.order**d for d, c in zip(degrees, coeffs)])

    p = poly.Roots(roots, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == degree
    assert np.array_equal(p.nonzero_degrees, nonzero_degrees)
    assert np.array_equal(p.nonzero_coeffs, nonzero_coeffs)
    assert np.array_equal(p.degrees, degrees)
    assert np.array_equal(p.coeffs, coeffs)
    assert p.integer == integer
