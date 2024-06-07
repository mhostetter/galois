"""
A pytest module to test Galois field polynomial instantiation.
"""

import random

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


@pytest.fixture(params=FIELDS)
def config(request):
    field = request.param
    d = {}
    d["GF"] = field
    d["degree"] = 5

    c1, c2, c3 = field.Random(low=1), field.Random(low=1), field.Random(low=1)
    d["nonzero_degrees"] = [5, 3, 0]
    d["nonzero_coeffs"] = [c1, c2, c3]

    d["degrees"] = [5, 4, 3, 2, 1, 0]
    d["coeffs"] = [c1, 0, c2, 0, 0, c3]
    d["integer"] = int(c1) * field.order**5 + int(c2) * field.order**3 + int(c3) * field.order**0
    s1, s2, s3 = int(c1) if c1 > 1 else "", int(c2) if c2 > 1 else "", c3
    d["string"] = f"{s1}x^5 + {s2}x^3 + {s3}"

    # Negative coefficients
    n1, n2, n3 = -abs(int(-c1)), -abs(int(-c2)), -abs(int(-c3))
    d["neg_coeffs"] = [n1, 0, n2, 0, 0, n3]
    d["neg_nonzero_coeffs"] = [n1, n2, n3]

    # Leading zeros
    d["lz_coeffs"] = [0, 0, c1, 0, c2, 0, 0, c3]

    # Mixed zeros
    d["mz_nonzero_degrees"] = [7, 6, 5, 3, 2, 0]
    d["mz_nonzero_coeffs"] = [0, 0, c1, c2, 0, c3]

    return d


def test_exceptions():
    with pytest.raises(TypeError):
        galois.Poly([1, 0, 1], field="invalid-type")
    with pytest.raises(TypeError):
        galois.Poly("invalid-type")
    with pytest.raises(ValueError):
        galois.Poly(np.array([[1, 0, 1], [1, 1, 1]]))
    with pytest.raises(ValueError):
        galois.Poly([1, 0, 1], order="invalid-type")


@pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
def test_coeffs(type1, config):
    GF = config["GF"]
    if type1 is not galois.FieldArray:
        p = galois.Poly(type1(config["coeffs"]), field=GF)
    else:
        p = galois.Poly(GF(config["coeffs"]))
    check_attributes(p, config)


@pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
def test_leading_zero_coeffs(type1, config):
    GF = config["GF"]
    if type1 is not galois.FieldArray:
        p = galois.Poly(type1(config["lz_coeffs"]), field=GF)
    else:
        p = galois.Poly(GF(config["lz_coeffs"]))
    check_attributes(p, config)


@pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
def test_ascending_coeffs(type1, config):
    GF = config["GF"]
    if type1 is not galois.FieldArray:
        p = galois.Poly(type1(config["coeffs"][::-1]), field=GF, order="asc")
    else:
        p = galois.Poly(GF(config["coeffs"][::-1]), order="asc")
    check_attributes(p, config)


@pytest.mark.parametrize("type1", [list, tuple, np.array])
def test_negative_coeffs(type1, config):
    GF = config["GF"]
    p = galois.Poly(type1(config["neg_coeffs"]), field=GF)
    check_attributes(p, config)


@pytest.mark.parametrize("type1", [list, tuple, np.array, galois.FieldArray])
@pytest.mark.parametrize("field", FIELDS)
def test_zero(type1, field):
    # The zero polynomial can cause problems, so test it explicitly
    if type1 is not galois.FieldArray:
        p = galois.Poly(type1([0]), field=field)
    else:
        p = galois.Poly(field([0]))
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 0
    assert np.array_equal(p.nonzero_degrees, [])
    assert np.array_equal(p.nonzero_coeffs, [])
    assert np.array_equal(p.degrees, [0])
    assert np.array_equal(p.coeffs, [0])
    assert int(p) == 0


@pytest.mark.parametrize("field", FIELDS)
def test_degree_zero(field):
    value = 0
    coeffs = field([value])
    p = galois.Poly(field(value))
    assert np.array_equal(p.coeffs, coeffs)
    p = galois.Poly(field([value]))
    assert np.array_equal(p.coeffs, coeffs)
    p = galois.Poly([value], field=field)
    assert np.array_equal(p.coeffs, coeffs)

    value = random.randint(1, field.order - 1)
    coeffs = field([value])
    p = galois.Poly(field(value))
    assert np.array_equal(p.coeffs, coeffs)
    p = galois.Poly(field([value]))
    assert np.array_equal(p.coeffs, coeffs)
    p = galois.Poly([value], field=field)
    assert np.array_equal(p.coeffs, coeffs)


def test_field_override():
    coeffs = [1, 0, 1, 1]
    p = galois.Poly(galois.GF2(coeffs))
    assert p.field == galois.GF2
    assert np.array_equal(p.coeffs, coeffs)

    GF = galois.GF(2**8)
    p = galois.Poly(galois.GF2(coeffs), field=GF)
    assert p.field == GF
    assert np.array_equal(p.coeffs, coeffs)


def check_attributes(poly, config):
    assert isinstance(poly, galois.Poly)
    assert poly.field is config["GF"]
    assert poly.degree == config["degree"]
    assert np.array_equal(poly.nonzero_degrees, config["nonzero_degrees"])
    assert np.array_equal(poly.nonzero_coeffs, config["nonzero_coeffs"])
    assert np.array_equal(poly.degrees, config["degrees"])
    assert np.array_equal(poly.coeffs, config["coeffs"])
    assert int(poly) == config["integer"]
    assert str(poly) == config["string"]
