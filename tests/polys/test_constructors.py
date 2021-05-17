"""
A pytest module to test Galois field polynomial alternate constructors.
"""
from typing import Type
import numpy as np
import pytest

import galois


FIELDS = [
    galois.GF2,  # GF(2)
    galois.GF(31),  # GF(p) with np.int dtypes
    galois.GF(36893488147419103183),  # GF(p) with object dtype
    galois.GF(2**8),  # GF(2^m) with np.int dtypes
    galois.GF(2**100),  # GF(2^m) with object dtype
    galois.GF(7**3),  # GF(p^m) with np.int dtypes
    galois.GF(109987**4),  # GF(p^m) with object dtypes
]


@pytest.mark.parametrize("field", FIELDS)
def test_zero(field):
    p = galois.Poly.Zero(field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 0
    assert np.array_equal(p.nonzero_degrees, [])
    assert np.array_equal(p.nonzero_coeffs, [])
    assert np.array_equal(p.degrees, [0])
    assert np.array_equal(p.coeffs, [0])
    assert p.integer == 0


@pytest.mark.parametrize("field", FIELDS)
def test_one(field):
    p = galois.Poly.One(field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 0
    assert np.array_equal(p.nonzero_degrees, [0])
    assert np.array_equal(p.nonzero_coeffs, [1])
    assert np.array_equal(p.degrees, [0])
    assert np.array_equal(p.coeffs, [1])
    assert p.integer == 1


@pytest.mark.parametrize("field", FIELDS)
def test_identity(field):
    p = galois.Poly.Identity(field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 1
    assert np.array_equal(p.nonzero_degrees, [1])
    assert np.array_equal(p.nonzero_coeffs, [1])
    assert np.array_equal(p.degrees, [1,0])
    assert np.array_equal(p.coeffs, [1,0])
    assert p.integer == field.order


@pytest.mark.parametrize("field", FIELDS)
def test_random(field):
    p = galois.Poly.Random(2, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2


def test_random_exceptions():
    with pytest.raises(TypeError):
        p = galois.Poly.Random(2.0)
    with pytest.raises(TypeError):
        p = galois.Poly.Random(-1)


@pytest.mark.parametrize("field", FIELDS)
def test_integer(field):
    integer = field.order + 1  # Corresponds to p(x) = x + 1
    p = galois.Poly.Integer(integer, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 1
    assert np.array_equal(p.nonzero_degrees, [1,0])
    assert np.array_equal(p.nonzero_coeffs, [1,1])
    assert np.array_equal(p.degrees, [1,0])
    assert np.array_equal(p.coeffs, [1,1])
    assert p.integer == integer


def test_integer_exceptions():
    with pytest.raises(TypeError):
        p = galois.Poly.Integer(5.0)


@pytest.mark.parametrize("field", FIELDS)
def test_degrees(field):
    # Corresponds to p(x) = x^2 + 1
    degrees = [2,0]
    coeffs = [1,1]
    p = galois.Poly.Degrees(degrees, coeffs, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2
    assert np.array_equal(p.nonzero_degrees, [2,0])
    assert np.array_equal(p.nonzero_coeffs, [1,1])
    assert np.array_equal(p.degrees, [2,1,0])
    assert np.array_equal(p.coeffs, [1,0,1])
    assert p.integer == field.order**2 + 1


@pytest.mark.parametrize("field", FIELDS)
def test_degrees_empty(field):
    p = galois.Poly.Degrees([], field=field)
    assert p == galois.Poly([0], field=field)


def test_degrees_exceptions():
    GF = galois.GF(3)
    degrees = [5, 3, 0]
    coeffs = [1, 2, 1]
    with pytest.raises(TypeError):
        galois.Poly.Degrees("invalid-type", coeffs=coeffs, field=GF)
    with pytest.raises(TypeError):
        galois.Poly.Degrees(degrees, coeffs="invalid-type", field=GF)
    with pytest.raises(ValueError):
        galois.Poly.Degrees([7] + degrees, coeffs=coeffs, field=GF)
    with pytest.raises(ValueError):
        galois.Poly.Degrees([5, -3, 0], coeffs=coeffs, field=GF)


@pytest.mark.parametrize("field", FIELDS)
def test_roots(field):
    a, b = field.Random(), field.Random()
    roots = [a, b]  # p(x) = (x - a)*(x - b)
    degree = 2
    degrees = [2, 1, 0]
    coeffs = [1, -a + -b, (-a)*(-b)]
    nonzero_degrees = [d for d, c in zip(degrees, coeffs) if c > 0]
    nonzero_coeffs = [c for d, c in zip(degrees, coeffs) if c > 0]
    integer = sum([int(c)*field.order**d for d, c in zip(degrees, coeffs)])

    p = galois.Poly.Roots(roots, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == degree
    assert np.array_equal(p.nonzero_degrees, nonzero_degrees)
    assert np.array_equal(p.nonzero_coeffs, nonzero_coeffs)
    assert np.array_equal(p.degrees, degrees)
    assert np.array_equal(p.coeffs, coeffs)
    assert p.integer == integer


@pytest.mark.parametrize("field", FIELDS)
def test_roots_with_multiplicity(field):
    a = field.Random()
    roots = [a]  # p(x) = (x - a)*(x - a)
    multiplicities = [2]
    degree = 2
    degrees = [2, 1, 0]
    coeffs = [1, -a + -a, (-a)*(-a)]
    nonzero_degrees = [d for d, c in zip(degrees, coeffs) if c > 0]
    nonzero_coeffs = [c for d, c in zip(degrees, coeffs) if c > 0]
    integer = sum([int(c)*field.order**d for d, c in zip(degrees, coeffs)])

    p = galois.Poly.Roots(roots, multiplicities=multiplicities, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == degree
    assert np.array_equal(p.nonzero_degrees, nonzero_degrees)
    assert np.array_equal(p.nonzero_coeffs, nonzero_coeffs)
    assert np.array_equal(p.degrees, degrees)
    assert np.array_equal(p.coeffs, coeffs)
    assert p.integer == integer


def test_roots_exceptions():
    GF = galois.GF(2**8)
    roots = [134, 212]
    multiplicities = [1, 2]
    with pytest.raises(TypeError):
        galois.Poly.Roots(roots, field="invalid-type")
    with pytest.raises(TypeError):
        galois.Poly.Roots(134, field=GF)
    with pytest.raises(TypeError):
        galois.Poly.Roots(roots, multiplicities=134, field=GF)
    with pytest.raises(ValueError):
        galois.Poly.Roots(roots, multiplicities=multiplicities + [1], field=GF)
