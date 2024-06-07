"""
A pytest module to test generating primitive polynomials and testing primitivity.

References
----------
* https://baylor-ir.tdl.org/bitstream/handle/2104/8793/GF3%20Polynomials.pdf?sequence=1&isAllowed=y
"""

import numpy as np
import pytest

import galois

from .luts.primitive_polys import PRIMITIVE_POLYS


def test_primitive_poly_exceptions():
    with pytest.raises(TypeError):
        galois.primitive_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.primitive_poly(2, 3.0)
    with pytest.raises(TypeError):
        galois.primitive_poly(2, 3, terms=2.0)

    with pytest.raises(ValueError):
        galois.primitive_poly(2**2 * 3**2, 3)
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 0)
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 3, terms=6)
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 3, terms="invalid-argument")
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 3, method="invalid-argument")


@pytest.mark.parametrize("order,degree,polys", PRIMITIVE_POLYS)
def test_primitive_poly_min(order, degree, polys):
    assert galois.primitive_poly(order, degree).coeffs.tolist() == polys[0]


@pytest.mark.parametrize("order,degree,polys", PRIMITIVE_POLYS)
def test_primitive_poly_max(order, degree, polys):
    assert galois.primitive_poly(order, degree, method="max").coeffs.tolist() == polys[-1]


@pytest.mark.parametrize("order,degree,polys", PRIMITIVE_POLYS)
def test_primitive_poly_random(order, degree, polys):
    assert galois.primitive_poly(order, degree, method="random").coeffs.tolist() in polys


def test_primitive_polys_exceptions():
    with pytest.raises(TypeError):
        next(galois.primitive_polys(2.0, 3))
    with pytest.raises(TypeError):
        next(galois.primitive_polys(2, 3.0))
    with pytest.raises(TypeError):
        next(galois.primitive_polys(2, 3, terms=2.0))
    with pytest.raises(TypeError):
        next(galois.primitive_polys(2, 3, reverse=1))

    with pytest.raises(ValueError):
        next(galois.primitive_polys(2**2 * 3**2, 3))
    with pytest.raises(ValueError):
        next(galois.primitive_polys(2, -1))
    with pytest.raises(ValueError):
        next(galois.primitive_polys(2, 3, terms=6))
    with pytest.raises(ValueError):
        next(galois.primitive_polys(2, 3, terms="invalid-argument"))


@pytest.mark.parametrize("order,degree,polys", PRIMITIVE_POLYS)
def test_primitive_polys(order, degree, polys):
    assert [f.coeffs.tolist() for f in galois.primitive_polys(order, degree)] == polys


@pytest.mark.parametrize("order,degree,polys", PRIMITIVE_POLYS)
def test_specific_terms(order, degree, polys):
    all_polys = []
    for terms in range(1, degree + 2):
        new_polys = list(galois.primitive_polys(order, degree, terms=terms))
        assert all(p.nonzero_coeffs.size == terms for p in new_polys)
        all_polys += new_polys
    all_polys = [p.coeffs.tolist() for p in sorted(all_polys, key=int)]
    assert all_polys == polys


def test_specific_terms_none_found():
    with pytest.raises(RuntimeError):
        galois.primitive_poly(2, 3, terms=2)

    assert not list(galois.primitive_polys(2, 3, terms=2))


@pytest.mark.parametrize("order,degree,polys", PRIMITIVE_POLYS)
def test_minimum_terms(order, degree, polys):
    min_terms = min(np.count_nonzero(f) for f in polys)
    min_term_polys = [f for f in polys if np.count_nonzero(f) == min_terms]
    assert [f.coeffs.tolist() for f in galois.primitive_polys(order, degree, terms="min")] == min_term_polys

    f = galois.primitive_poly(order, degree, terms="min", method="random")
    assert f.coeffs.tolist() in min_term_polys


def test_matlab_primitive_poly_exceptions():
    with pytest.raises(TypeError):
        galois.matlab_primitive_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.matlab_primitive_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.matlab_primitive_poly(4, 3)
    with pytest.raises(ValueError):
        galois.matlab_primitive_poly(2, 0)


def test_matlab_primitive_poly():
    """
    Matlab:
        % Note Matlab's ordering is degree-ascending
        gfprimdf(m, p)
    """
    assert galois.matlab_primitive_poly(2, 1).coeffs.tolist()[::-1] == [1, 1]
    assert galois.matlab_primitive_poly(2, 2).coeffs.tolist()[::-1] == [1, 1, 1]
    assert galois.matlab_primitive_poly(2, 3).coeffs.tolist()[::-1] == [1, 1, 0, 1]
    assert galois.matlab_primitive_poly(2, 4).coeffs.tolist()[::-1] == [1, 1, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 5).coeffs.tolist()[::-1] == [1, 0, 1, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 6).coeffs.tolist()[::-1] == [1, 1, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 7).coeffs.tolist()[::-1] == [1, 0, 0, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 8).coeffs.tolist()[::-1] == [1, 0, 1, 1, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 9).coeffs.tolist()[::-1] == [1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 10).coeffs.tolist()[::-1] == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 11).coeffs.tolist()[::-1] == [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 12).coeffs.tolist()[::-1] == [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 13).coeffs.tolist()[::-1] == [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 14).coeffs.tolist()[::-1] == [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 15).coeffs.tolist()[::-1] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 16).coeffs.tolist()[::-1] == [
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    ]

    assert galois.matlab_primitive_poly(3, 1).coeffs.tolist()[::-1] == [1, 1]
    assert galois.matlab_primitive_poly(3, 2).coeffs.tolist()[::-1] == [2, 1, 1]
    assert galois.matlab_primitive_poly(3, 3).coeffs.tolist()[::-1] == [1, 2, 0, 1]
    assert galois.matlab_primitive_poly(3, 4).coeffs.tolist()[::-1] == [2, 1, 0, 0, 1]
    assert galois.matlab_primitive_poly(3, 5).coeffs.tolist()[::-1] == [1, 2, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(3, 6).coeffs.tolist()[::-1] == [2, 1, 0, 0, 0, 0, 1]
    # assert galois.matlab_primitive_poly(3, 7).coeffs.tolist()[::-1] == [1, 0, 2, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(3, 8).coeffs.tolist()[::-1] == [2, 0, 0, 1, 0, 0, 0, 0, 1]

    assert galois.matlab_primitive_poly(5, 1).coeffs.tolist()[::-1] == [2, 1]
    assert galois.matlab_primitive_poly(5, 2).coeffs.tolist()[::-1] == [2, 1, 1]
    assert galois.matlab_primitive_poly(5, 3).coeffs.tolist()[::-1] == [2, 3, 0, 1]
    # assert galois.matlab_primitive_poly(5, 4).coeffs.tolist()[::-1] == [2, 2, 1, 0, 1]
    assert galois.matlab_primitive_poly(5, 5).coeffs.tolist()[::-1] == [2, 4, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(5, 6).coeffs.tolist()[::-1] == [2, 1, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(5, 7).coeffs.tolist()[::-1] == [2, 3, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(5, 8).coeffs.tolist()[::-1] == [3, 2, 1, 0, 0, 0, 0, 0, 1]

    assert galois.matlab_primitive_poly(7, 1).coeffs.tolist()[::-1] == [2, 1]
    assert galois.matlab_primitive_poly(7, 2).coeffs.tolist()[::-1] == [3, 1, 1]
    assert galois.matlab_primitive_poly(7, 3).coeffs.tolist()[::-1] == [2, 3, 0, 1]
    assert galois.matlab_primitive_poly(7, 4).coeffs.tolist()[::-1] == [5, 3, 1, 0, 1]
    assert galois.matlab_primitive_poly(7, 5).coeffs.tolist()[::-1] == [4, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(7, 6).coeffs.tolist()[::-1] == [5, 1, 3, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(7, 7).coeffs.tolist()[::-1] == [2, 6, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(7, 8).coeffs.tolist()[::-1] == [3, 1, 0, 0, 0, 0, 0, 0, 1]
