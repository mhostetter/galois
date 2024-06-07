"""
A pytest module to test Galois field polynomial alternate constructors.
"""

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


def test_zero_exceptions():
    with pytest.raises(TypeError):
        galois.Poly.Zero(field=type(galois.Array))


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
    assert int(p) == 0
    assert str(p) == "0"


def test_one_exceptions():
    with pytest.raises(TypeError):
        galois.Poly.One(field=type(galois.Array))


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
    assert int(p) == 1
    assert str(p) == "1"


def test_identity_exceptions():
    with pytest.raises(TypeError):
        galois.Poly.Identity(field=type(galois.Array))


@pytest.mark.parametrize("field", FIELDS)
def test_identity(field):
    p = galois.Poly.Identity(field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 1
    assert np.array_equal(p.nonzero_degrees, [1])
    assert np.array_equal(p.nonzero_coeffs, [1])
    assert np.array_equal(p.degrees, [1, 0])
    assert np.array_equal(p.coeffs, [1, 0])
    assert int(p) == field.order
    assert str(p) == "x"


def test_random_exceptions():
    with pytest.raises(TypeError):
        galois.Poly.Random(2.0)
    with pytest.raises(TypeError):
        galois.Poly.Random(2, field=type(galois.Array))
    with pytest.raises(TypeError):
        galois.Poly.Random(2, seed=3.14)
    with pytest.raises(ValueError):
        galois.Poly.Random(-1)
    with pytest.raises(ValueError):
        galois.Poly.Random(2, seed=-1)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("seed", [None, 42, np.int64(1337), np.ulonglong(27182818284), np.random.default_rng(123456)])
def test_random(field, seed):
    p = galois.Poly.Random(2, field=field, seed=seed)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2


def test_integer_exceptions():
    with pytest.raises(TypeError):
        galois.Poly.Int(5.0)
    with pytest.raises(TypeError):
        galois.Poly.Int(5, field=type(galois.Array))
    with pytest.raises(ValueError):
        galois.Poly.Int(-1)


@pytest.mark.parametrize("field", FIELDS)
def test_integer(field):
    integer = field.order + 1  # Corresponds to p(x) = x + 1
    p = galois.Poly.Int(integer, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 1
    assert np.array_equal(p.nonzero_degrees, [1, 0])
    assert np.array_equal(p.nonzero_coeffs, [1, 1])
    assert np.array_equal(p.degrees, [1, 0])
    assert np.array_equal(p.coeffs, [1, 1])
    assert int(p) == integer


def test_string_exceptions():
    with pytest.raises(TypeError):
        galois.Poly.Str(b"x + 5")
    with pytest.raises(TypeError):
        galois.Poly.Str("x + 5", field=type(galois.Array))


@pytest.mark.parametrize("field", FIELDS)
def test_string(field):
    degrees = [10, 7, 4, 1, 0]
    coeffs = [field.Random(low=1), field.Random(low=1), field.Random(low=1), field.Random(low=1), field.Random(low=1)]
    string = create_string(coeffs, degrees)

    p = galois.Poly.Str(string, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == max(degrees)
    assert np.array_equal(p.nonzero_degrees, degrees)
    assert np.array_equal(p.nonzero_coeffs, coeffs)
    assert str(p) == create_string(coeffs, degrees)

    degrees = [105, 97, 48, 1, 0]
    coeffs = [field.Random(low=1), field.Random(low=1), field.Random(low=1), field.Random(low=1), field.Random(low=1)]
    string = create_string(coeffs, degrees)

    p = galois.Poly.Str(string, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == max(degrees)
    assert np.array_equal(p.nonzero_degrees, degrees)
    assert np.array_equal(p.nonzero_coeffs, coeffs)
    assert str(p) == string


def test_string_large():
    string = "x^106 + x^105 + x^104 + x^103 + x^102 + x^101 + x^100 + x^99 + x^98 + x^97 + x^96 + x^95 + x^94 + x^93 + x^92 + x^91 + x^90 + x^89 + x^88 + x^87 + x^86 + x^85 + x^84 + x^83 + x^82 + x^81 + x^80 + x^79 + x^78 + x^77 + x^76 + x^75 + x^74 + x^73 + x^72 + x^71 + x^70 + x^69 + x^68 + x^67 + x^66 + x^65 + x^64 + x^63 + x^62 + x^61 + x^60 + x^59 + x^58 + x^57 + x^56 + x^55 + x^54 + x^53 + x^52 + x^51 + x^50 + x^49 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^42 + x^41 + x^40 + x^39 + x^38 + x^37 + x^36 + x^35 + x^34 + x^33 + x^32 + x^31 + x^30 + x^29 + x^28 + x^27 + x^26 + x^25 + x^24 + x^23 + x^22 + x^21 + x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + x^12 + x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2"
    p = galois.Poly.Str(string)
    assert isinstance(p, galois.Poly)
    assert p.field is galois.GF2
    assert str(p) == string


def test_degrees_exceptions():
    GF = galois.GF(3)
    degrees = [5, 3, 0]
    coeffs = [1, 2, 1]

    with pytest.raises(TypeError):
        galois.Poly.Degrees("invalid-type", coeffs=coeffs, field=GF)
    with pytest.raises(TypeError):
        galois.Poly.Degrees(degrees, coeffs="invalid-type", field=GF)
    with pytest.raises(TypeError):
        galois.Poly.Degrees(degrees, coeffs=coeffs, field=type(galois.Array))

    with pytest.raises(ValueError):
        galois.Poly.Degrees(np.atleast_2d(degrees), coeffs=coeffs, field=GF)
    with pytest.raises(ValueError):
        galois.Poly.Degrees([5, 1, 1, 0], coeffs=[1, 2, 2, 1], field=GF)
    with pytest.raises(ValueError):
        galois.Poly.Degrees([5, -3, 0], coeffs=coeffs, field=GF)
    with pytest.raises(ValueError):
        galois.Poly.Degrees(degrees, coeffs=np.atleast_2d(coeffs), field=GF)
    with pytest.raises(ValueError):
        galois.Poly.Degrees([7] + degrees, coeffs=coeffs, field=GF)


@pytest.mark.parametrize("field", FIELDS)
def test_degrees(field):
    # Corresponds to p(x) = x^2 + 1
    degrees = [2, 0]
    coeffs = [1, 1]

    p = galois.Poly.Degrees(degrees, coeffs, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2
    assert np.array_equal(p.nonzero_degrees, [2, 0])
    assert np.array_equal(p.nonzero_coeffs, [1, 1])
    assert np.array_equal(p.degrees, [2, 1, 0])
    assert np.array_equal(p.coeffs, [1, 0, 1])
    assert int(p) == field.order**2 + 1
    assert str(p) == "x^2 + 1"

    p = galois.Poly.Degrees(degrees, field(coeffs))
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == 2
    assert np.array_equal(p.nonzero_degrees, [2, 0])
    assert np.array_equal(p.nonzero_coeffs, [1, 1])
    assert np.array_equal(p.degrees, [2, 1, 0])
    assert np.array_equal(p.coeffs, [1, 0, 1])
    assert int(p) == field.order**2 + 1
    assert str(p) == "x^2 + 1"


@pytest.mark.parametrize("field", FIELDS)
def test_degrees_empty(field):
    p = galois.Poly.Degrees([], field=field)
    assert p == galois.Poly([0], field=field)
    assert p.field is field
    assert p.degree == 0
    assert np.array_equal(p.nonzero_degrees, [])
    assert np.array_equal(p.nonzero_coeffs, [])
    assert np.array_equal(p.degrees, [0])
    assert np.array_equal(p.coeffs, [0])
    assert int(p) == 0
    assert str(p) == "0"


def test_roots_exceptions():
    GF = galois.GF(2**8)
    roots = [134, 212]
    multiplicities = [1, 2]

    with pytest.raises(TypeError):
        galois.Poly.Roots(134, field=GF)
    with pytest.raises(TypeError):
        galois.Poly.Roots(roots, field="invalid-type")
    with pytest.raises(TypeError):
        galois.Poly.Roots(roots, multiplicities=134, field=GF)

    with pytest.raises(ValueError):
        galois.Poly.Roots(roots, multiplicities=multiplicities + [1], field=GF)


@pytest.mark.parametrize("field", FIELDS)
def test_roots(field):
    a, b = field.Random(), field.Random()
    roots = [a, b]  # p(x) = (x - a)*(x - b)
    degree = 2
    degrees = [2, 1, 0]
    coeffs = [1, -a + -b, (-a) * (-b)]
    nonzero_degrees = [d for d, c in zip(degrees, coeffs) if c > 0]
    nonzero_coeffs = [c for d, c in zip(degrees, coeffs) if c > 0]
    integer = sum(int(c) * field.order**d for d, c in zip(degrees, coeffs))

    p = galois.Poly.Roots(roots, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == degree
    assert np.array_equal(p.nonzero_degrees, nonzero_degrees)
    assert np.array_equal(p.nonzero_coeffs, nonzero_coeffs)
    assert np.array_equal(p.degrees, degrees)
    assert np.array_equal(p.coeffs, coeffs)
    assert int(p) == integer

    p = galois.Poly.Roots(field(roots))
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == degree
    assert np.array_equal(p.nonzero_degrees, nonzero_degrees)
    assert np.array_equal(p.nonzero_coeffs, nonzero_coeffs)
    assert np.array_equal(p.degrees, degrees)
    assert np.array_equal(p.coeffs, coeffs)
    assert int(p) == integer


@pytest.mark.parametrize("field", FIELDS)
def test_roots_with_multiplicity(field):
    a = field.Random()
    roots = [a]  # p(x) = (x - a)*(x - a)
    multiplicities = [2]
    degree = 2
    degrees = [2, 1, 0]
    coeffs = [1, -a + -a, (-a) * (-a)]
    nonzero_degrees = [d for d, c in zip(degrees, coeffs) if c > 0]
    nonzero_coeffs = [c for d, c in zip(degrees, coeffs) if c > 0]
    integer = sum(int(c) * field.order**d for d, c in zip(degrees, coeffs))

    p = galois.Poly.Roots(roots, multiplicities=multiplicities, field=field)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == degree
    assert np.array_equal(p.nonzero_degrees, nonzero_degrees)
    assert np.array_equal(p.nonzero_coeffs, nonzero_coeffs)
    assert np.array_equal(p.degrees, degrees)
    assert np.array_equal(p.coeffs, coeffs)
    assert int(p) == integer

    p = galois.Poly.Roots(field(roots), multiplicities=multiplicities)
    assert isinstance(p, galois.Poly)
    assert p.field is field
    assert p.degree == degree
    assert np.array_equal(p.nonzero_degrees, nonzero_degrees)
    assert np.array_equal(p.nonzero_coeffs, nonzero_coeffs)
    assert np.array_equal(p.degrees, degrees)
    assert np.array_equal(p.coeffs, coeffs)
    assert int(p) == integer


def test_roots_field_override():
    roots = [1, 0]
    p = galois.Poly.Roots(galois.GF2(roots))
    assert p.field == galois.GF2

    GF = galois.GF(2**8)
    p = galois.Poly.Roots(galois.GF2(roots), field=GF)
    assert p.field == GF


def create_string(coeffs, degrees):
    string = " + ".join([f"{c if c > 1 else ''}x^{d}" for d, c in zip(degrees[:-2], coeffs[:-2])])
    string += f" + {coeffs[-2] if coeffs[-2] > 1 else ''}x"
    string += f" + {coeffs[-1] if coeffs[-1] > 0 else ''}"
    return string
