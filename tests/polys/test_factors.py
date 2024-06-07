"""
A pytest module to test factoring polynomials over finite fields.
"""

import random

import pytest

import galois

from .luts.poly_factors import POLY_FACTORS


def test_factors_exceptions():
    GF = galois.GF(5)
    with pytest.raises(TypeError):
        galois.factors([1, 0, 2, 4])
    with pytest.raises(ValueError):
        galois.factors(galois.Poly([2, 0, 2, 4], field=GF))
    with pytest.raises(ValueError):
        galois.factors(galois.Poly([2], field=GF))

    with pytest.raises(ValueError):
        galois.Poly([2, 0, 2, 4], field=GF).factors()
    with pytest.raises(ValueError):
        galois.Poly([2], field=GF).factors()


def test_factors_old():
    g0, g1, g2 = galois.conway_poly(2, 3), galois.conway_poly(2, 4), galois.conway_poly(2, 5)
    k0, k1, k2 = 2, 3, 4
    f = g0**k0 * g1**k1 * g2**k2
    factors, multiplicities = galois.factors(f) if random.choice([True, False]) else f.factors()
    assert factors == [g0, g1, g2]
    assert multiplicities == [k0, k1, k2]

    g0, g1, g2 = galois.conway_poly(3, 3), galois.conway_poly(3, 4), galois.conway_poly(3, 5)
    k0, k1, k2 = 3, 4, 6
    f = g0**k0 * g1**k1 * g2**k2
    factors, multiplicities = galois.factors(f)
    assert factors == [g0, g1, g2]
    assert multiplicities == [k0, k1, k2]


def test_factors_random():
    for _ in range(5):
        f = galois.Poly.Random(random.randint(10, 50))
        factors, multiplicities = galois.factors(f) if random.choice([True, False]) else f.factors()
        assert f == multiply_factors(factors, multiplicities)

    GF = galois.GF(5)
    for _ in range(5):
        f = galois.Poly.Random(random.randint(10, 50), field=GF)
        f //= f.coeffs[0]  # Make monic
        factors, multiplicities = galois.factors(f) if random.choice([True, False]) else f.factors()
        assert f == multiply_factors(factors, multiplicities)


@pytest.mark.parametrize("characteristic,degree,lut", POLY_FACTORS)
def test_factors(characteristic, degree, lut):
    GF = galois.GF(characteristic**degree)

    for item in lut:
        a = galois.Poly(item[0], field=GF)
        factors = [galois.Poly(f, field=GF) for f in item[1]]
        multiplicities = item[2]

        # Sort the Sage output to be ordered similarly to `galois`
        factors, multiplicities = zip(*sorted(zip(factors, multiplicities), key=lambda item: int(item[0])))
        factors, multiplicities = list(factors), list(multiplicities)

        if random.choice([True, False]):
            assert galois.factors(a) == (factors, multiplicities)
        else:
            assert a.factors() == (factors, multiplicities)


def test_square_free_factors_exceptions():
    GF = galois.GF(5)
    with pytest.raises(ValueError):
        galois.Poly([2, 0, 2, 4], field=GF).square_free_factors()
    with pytest.raises(ValueError):
        galois.Poly([2], field=GF).square_free_factors()


def test_square_free_factors():
    a = galois.irreducible_poly(2, 1, method="random")
    b = galois.irreducible_poly(2, 4, method="random")
    c = galois.irreducible_poly(2, 3, method="random")
    f = a * b * c**3
    assert f.square_free_factors() == ([a * b, c], [1, 3])

    a = galois.irreducible_poly(5, 1, method="random")
    b = galois.irreducible_poly(5, 4, method="random")
    c = galois.irreducible_poly(5, 3, method="random")
    f = a * b * c**3
    assert f.square_free_factors() == ([a * b, c], [1, 3])

    a = galois.irreducible_poly(2**2, 1, method="random")
    b = galois.irreducible_poly(2**2, 4, method="random")
    c = galois.irreducible_poly(2**2, 3, method="random")
    f = a * b * c**3
    assert f.square_free_factors() == ([a * b, c], [1, 3])

    a = galois.irreducible_poly(5**2, 1, method="random")
    b = galois.irreducible_poly(5**2, 4, method="random")
    c = galois.irreducible_poly(5**2, 3, method="random")
    f = a * b * c**3
    assert f.square_free_factors() == ([a * b, c], [1, 3])


def test_square_free_factors_random():
    GF = galois.GF(2)
    f = galois.Poly.Random(10, field=GF)
    f //= f.coeffs[0]  # Make monic
    factors, multiplicities = f.square_free_factors()
    assert f == multiply_factors(factors, multiplicities)

    GF = galois.GF(5)
    f = galois.Poly.Random(10, field=GF)
    f //= f.coeffs[0]  # Make monic
    factors, multiplicities = f.square_free_factors()
    assert f == multiply_factors(factors, multiplicities)

    GF = galois.GF(2**2)
    f = galois.Poly.Random(10, field=GF)
    f //= f.coeffs[0]  # Make monic
    factors, multiplicities = f.square_free_factors()
    assert f == multiply_factors(factors, multiplicities)

    GF = galois.GF(5**2)
    f = galois.Poly.Random(10, field=GF)
    f //= f.coeffs[0]  # Make monic
    factors, multiplicities = f.square_free_factors()
    assert f == multiply_factors(factors, multiplicities)


def test_distinct_degree_factors_exceptions():
    GF = galois.GF(5)
    with pytest.raises(ValueError):
        galois.Poly([2, 0, 2, 4], field=GF).distinct_degree_factors()
    with pytest.raises(ValueError):
        galois.Poly([2], field=GF).distinct_degree_factors()


def test_distinct_degree_factors():
    GF = galois.GF(2)
    f, factors, degrees = create_distinct_degree_poly(GF, [2, 2, 3])
    assert f.distinct_degree_factors() == (factors, degrees)

    GF = galois.GF(5)
    f, factors, degrees = create_distinct_degree_poly(GF, [5, 5, 5])
    assert f.distinct_degree_factors() == (factors, degrees)

    GF = galois.GF(2**2)
    f, factors, degrees = create_distinct_degree_poly(GF, [3, 5, 5])
    assert f.distinct_degree_factors() == (factors, degrees)

    GF = galois.GF(5**2)
    f, factors, degrees = create_distinct_degree_poly(GF, [3, 5, 5])
    assert f.distinct_degree_factors() == (factors, degrees)


def test_equal_degree_factors_exceptions():
    GF = galois.GF(5)
    a = galois.Poly([1, 0, 2, 1], field=GF)
    b = galois.Poly([1, 4, 4, 4], field=GF)
    f = a * b

    with pytest.raises(TypeError):
        f.equal_degree_factors(2.0)
    with pytest.raises(ValueError):
        galois.Poly([2], field=GF).equal_degree_factors(1)
    with pytest.raises(ValueError):
        galois.Poly([2, 0, 2, 4], field=GF).equal_degree_factors(2)
    with pytest.raises(ValueError):
        f.equal_degree_factors(4)


def test_equal_degree_factors():
    GF = galois.GF(2)
    f, d, factors = create_equal_degree_poly(GF, [5, 8], [2, 4])
    assert f.equal_degree_factors(d) == factors

    GF = galois.GF(5)
    f, d, factors = create_equal_degree_poly(GF, [1, 3], [2, 4])
    assert f.equal_degree_factors(d) == factors

    GF = galois.GF(2**2)
    f, d, factors = create_equal_degree_poly(GF, [1, 4], [2, 4])
    assert f.equal_degree_factors(d) == factors

    GF = galois.GF(5**2)
    f, d, factors = create_equal_degree_poly(GF, [1, 2], [2, 4])
    assert f.equal_degree_factors(d) == factors


def multiply_factors(factors, multiplicities):
    g = galois.Poly.One(factors[0].field)
    for fi, mi in zip(factors, multiplicities):
        g *= fi**mi
    return g


def create_distinct_degree_poly(GF, max_factors):
    """
    Create a polynomial that is a product of irreducible polynomials with degrees 1, 3, and 4.
    """
    f1, factors_1 = galois.Poly.One(GF), []
    while len(factors_1) < random.randint(1, max_factors[0]):
        f = galois.irreducible_poly(GF.order, 1, method="random")
        if f not in factors_1:
            factors_1.append(f)
            f1 *= f

    f3, factors_3 = galois.Poly.One(GF), []
    while len(factors_3) < random.randint(1, max_factors[1]):
        f = galois.irreducible_poly(GF.order, 3, method="random")
        if f not in factors_3:
            factors_3.append(f)
            f3 *= f

    f4, factors_4 = galois.Poly.One(GF), []
    while len(factors_4) < random.randint(1, max_factors[2]):
        f = galois.irreducible_poly(GF.order, 4, method="random")
        if f not in factors_4:
            factors_4.append(f)
            f4 *= f

    factors = [f1, f3, f4]
    degrees = [1, 3, 4]
    f = f1 * f3 * f4

    return f, factors, degrees


def create_equal_degree_poly(GF, degree_range, factor_range):
    """
    Create a polynomial that is a product of `r` irreducible polynomials with degrees `d`.
    """
    d = random.randint(*degree_range)  # The degree of the factors
    r = random.randint(*factor_range)  # The number of factors
    f, factors = galois.Poly.One(GF), []

    while len(factors) < r:
        fi = galois.irreducible_poly(GF.order, d, method="random")
        if fi not in factors:
            factors.append(fi)
            f *= fi
    factors = sorted(factors, key=int)

    return f, d, factors
