"""
A pytest module to test generating irreducible polynomials over finite fields.
"""

import time

import numpy as np
import pytest

import galois

from .luts.irreducible_polys import IRREDUCIBLE_POLYS
from .luts.irreducible_polys_min import IRREDUCIBLE_POLYS_MIN


def test_irreducible_poly_exceptions():
    with pytest.raises(TypeError):
        galois.irreducible_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.irreducible_poly(2, 3.0)
    with pytest.raises(TypeError):
        galois.irreducible_poly(2, 3, terms=2.0)

    with pytest.raises(ValueError):
        galois.irreducible_poly(2**2 * 3**2, 3)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 0)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 3, terms=6)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 3, terms="invalid-argument")
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 3, method="invalid-argument")


@pytest.mark.parametrize("order,degree,polys", IRREDUCIBLE_POLYS)
def test_irreducible_poly_min(order, degree, polys):
    assert galois.irreducible_poly(order, degree).coeffs.tolist() == polys[0]


@pytest.mark.parametrize("order,degree,polys", IRREDUCIBLE_POLYS)
def test_irreducible_poly_max(order, degree, polys):
    assert galois.irreducible_poly(order, degree, method="max").coeffs.tolist() == polys[-1]


@pytest.mark.parametrize("order,degree,polys", IRREDUCIBLE_POLYS)
def test_irreducible_poly_random(order, degree, polys):
    assert galois.irreducible_poly(order, degree, method="random").coeffs.tolist() in polys


def test_irreducible_polys_exceptions():
    with pytest.raises(TypeError):
        next(galois.irreducible_polys(2.0, 3))
    with pytest.raises(TypeError):
        next(galois.irreducible_polys(2, 3.0))
    with pytest.raises(TypeError):
        next(galois.irreducible_polys(2, 3, terms=2.0))
    with pytest.raises(TypeError):
        next(galois.irreducible_polys(2, 3, reverse=1))

    with pytest.raises(ValueError):
        next(galois.irreducible_polys(2**2 * 3**2, 3))
    with pytest.raises(ValueError):
        next(galois.irreducible_polys(2, -1))
    with pytest.raises(ValueError):
        next(galois.irreducible_polys(2, 3, terms=6))
    with pytest.raises(ValueError):
        next(galois.irreducible_polys(2, 3, terms="invalid-argument"))


@pytest.mark.parametrize("order,degree,polys", IRREDUCIBLE_POLYS)
def test_irreducible_polys(order, degree, polys):
    assert [f.coeffs.tolist() for f in galois.irreducible_polys(order, degree)] == polys


@pytest.mark.parametrize("order,degree,polys", IRREDUCIBLE_POLYS)
def test_specific_terms(order, degree, polys):
    all_polys = []
    for terms in range(1, degree + 2):
        new_polys = list(galois.irreducible_polys(order, degree, terms=terms))
        assert all(p.nonzero_coeffs.size == terms for p in new_polys)
        all_polys += new_polys
    all_polys = [p.coeffs.tolist() for p in sorted(all_polys, key=int)]
    assert all_polys == polys


def test_specific_terms_none_found():
    with pytest.raises(RuntimeError):
        galois.irreducible_poly(2, 3, terms=2)

    assert not list(galois.irreducible_polys(2, 3, terms=2))


@pytest.mark.parametrize("order,degree,polys", IRREDUCIBLE_POLYS)
def test_minimum_terms(order, degree, polys):
    min_terms = min(np.count_nonzero(f) for f in polys)
    min_term_polys = [f for f in polys if np.count_nonzero(f) == min_terms]
    assert [f.coeffs.tolist() for f in galois.irreducible_polys(order, degree, terms="min")] == min_term_polys

    f = galois.irreducible_poly(order, degree, terms="min", method="random")
    assert f.coeffs.tolist() in min_term_polys


@pytest.mark.parametrize("order,degree,polys", IRREDUCIBLE_POLYS_MIN)
def test_minimum_terms_from_database(order, degree, polys):
    tick = time.time()
    p = galois.irreducible_poly(order, degree, terms="min")
    tock = time.time()
    assert tock - tick < 1.0
    db_degrees = p.nonzero_degrees.tolist()
    db_coeffs = p.nonzero_coeffs.tolist()
    exp_degrees, exp_coeffs = polys
    assert db_degrees == exp_degrees and db_coeffs == exp_coeffs


def test_issue_360():
    """
    See https://github.com/mhostetter/galois/issues/360.
    """
    f = galois.Poly.Degrees([233, 74, 0])
    assert f.is_irreducible()


def test_issue_575():
    """
    See https://github.com/mhostetter/galois/issues/575.
    """
    p = 2**127 - 1
    coeffs = [132937, -281708, 210865, -132177, 154492, -119403, 64244, -21729, 12062, -425, 325, 525, 110, 20, 4, 1, 1]
    GF = galois.GF(p)
    f = galois.Poly(coeffs[::-1], GF)
    assert f.is_irreducible()
