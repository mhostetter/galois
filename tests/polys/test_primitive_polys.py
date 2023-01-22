"""
A pytest module to test generating primitive polynomials and testing primitivity.

References
----------
* https://baylor-ir.tdl.org/bitstream/handle/2104/8793/GF3%20Polynomials.pdf?sequence=1&isAllowed=y
"""
import numpy as np
import pytest

import galois

from .luts.primitive_polys_2 import (
    PRIMITIVE_POLYS_2_1,
    PRIMITIVE_POLYS_2_2,
    PRIMITIVE_POLYS_2_3,
    PRIMITIVE_POLYS_2_4,
    PRIMITIVE_POLYS_2_5,
    PRIMITIVE_POLYS_2_6,
    PRIMITIVE_POLYS_2_7,
    PRIMITIVE_POLYS_2_8,
)
from .luts.primitive_polys_3 import (
    PRIMITIVE_POLYS_3_1,
    PRIMITIVE_POLYS_3_2,
    PRIMITIVE_POLYS_3_3,
    PRIMITIVE_POLYS_3_4,
    PRIMITIVE_POLYS_3_5,
    PRIMITIVE_POLYS_3_6,
)
from .luts.primitive_polys_4 import (
    PRIMITIVE_POLYS_4_1,
    PRIMITIVE_POLYS_4_2,
    PRIMITIVE_POLYS_4_3,
)
from .luts.primitive_polys_5 import (
    PRIMITIVE_POLYS_5_1,
    PRIMITIVE_POLYS_5_2,
    PRIMITIVE_POLYS_5_3,
    PRIMITIVE_POLYS_5_4,
)
from .luts.primitive_polys_9 import (
    PRIMITIVE_POLYS_9_1,
    PRIMITIVE_POLYS_9_2,
    PRIMITIVE_POLYS_9_3,
)
from .luts.primitive_polys_25 import PRIMITIVE_POLYS_25_1, PRIMITIVE_POLYS_25_2

from .luts.primitive_polys_database import (
    PRIMITIVE_POLY_MIN_TERMS_2_723,
    PRIMITIVE_POLY_MIN_TERMS_2_936,
    PRIMITIVE_POLY_MIN_TERMS_2_1056,
    PRIMITIVE_POLY_MIN_TERMS_2_1200,
    PRIMITIVE_POLY_MIN_TERMS_3_399,
    PRIMITIVE_POLY_MIN_TERMS_3_422,
    PRIMITIVE_POLY_MIN_TERMS_3_549,
    PRIMITIVE_POLY_MIN_TERMS_3_660,
    PRIMITIVE_POLY_MIN_TERMS_5_296,
    PRIMITIVE_POLY_MIN_TERMS_5_357,
    PRIMITIVE_POLY_MIN_TERMS_5_404,
    PRIMITIVE_POLY_MIN_TERMS_5_430,
    PRIMITIVE_POLY_MIN_TERMS_7_129,
    PRIMITIVE_POLY_MIN_TERMS_7_195,
    PRIMITIVE_POLY_MIN_TERMS_7_284,
    PRIMITIVE_POLY_MIN_TERMS_7_357,
)

PARAMS = [
    (2, 1, PRIMITIVE_POLYS_2_1),
    (2, 2, PRIMITIVE_POLYS_2_2),
    (2, 3, PRIMITIVE_POLYS_2_3),
    (2, 4, PRIMITIVE_POLYS_2_4),
    (2, 5, PRIMITIVE_POLYS_2_5),
    (2, 6, PRIMITIVE_POLYS_2_6),
    (2, 7, PRIMITIVE_POLYS_2_7),
    (2, 8, PRIMITIVE_POLYS_2_8),
    (2**2, 1, PRIMITIVE_POLYS_4_1),
    (2**2, 2, PRIMITIVE_POLYS_4_2),
    (2**2, 3, PRIMITIVE_POLYS_4_3),
    (3, 1, PRIMITIVE_POLYS_3_1),
    (3, 2, PRIMITIVE_POLYS_3_2),
    (3, 3, PRIMITIVE_POLYS_3_3),
    (3, 4, PRIMITIVE_POLYS_3_4),
    (3, 5, PRIMITIVE_POLYS_3_5),
    (3, 6, PRIMITIVE_POLYS_3_6),
    (3**2, 1, PRIMITIVE_POLYS_9_1),
    (3**2, 2, PRIMITIVE_POLYS_9_2),
    (3**2, 3, PRIMITIVE_POLYS_9_3),
    (5, 1, PRIMITIVE_POLYS_5_1),
    (5, 2, PRIMITIVE_POLYS_5_2),
    (5, 3, PRIMITIVE_POLYS_5_3),
    (5, 4, PRIMITIVE_POLYS_5_4),
    (5**2, 1, PRIMITIVE_POLYS_25_1),
    (5**2, 2, PRIMITIVE_POLYS_25_2),
]

PARAMS_DB = [
    (2, 723, PRIMITIVE_POLY_MIN_TERMS_2_723),
    (2, 936, PRIMITIVE_POLY_MIN_TERMS_2_936),
    (2, 1056, PRIMITIVE_POLY_MIN_TERMS_2_1056),
    (2, 1200, PRIMITIVE_POLY_MIN_TERMS_2_1200),
    (3, 399, PRIMITIVE_POLY_MIN_TERMS_3_399),
    (3, 422, PRIMITIVE_POLY_MIN_TERMS_3_422),
    (3, 549, PRIMITIVE_POLY_MIN_TERMS_3_549),
    (3, 660, PRIMITIVE_POLY_MIN_TERMS_3_660),
    (5, 296, PRIMITIVE_POLY_MIN_TERMS_5_296),
    (5, 357, PRIMITIVE_POLY_MIN_TERMS_5_357),
    (5, 404, PRIMITIVE_POLY_MIN_TERMS_5_404),
    (5, 430, PRIMITIVE_POLY_MIN_TERMS_5_430),
    (7, 129, PRIMITIVE_POLY_MIN_TERMS_7_129),
    (7, 195, PRIMITIVE_POLY_MIN_TERMS_7_195),
    (7, 284, PRIMITIVE_POLY_MIN_TERMS_7_284),
    (7, 357, PRIMITIVE_POLY_MIN_TERMS_7_357),
]


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


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_primitive_poly_min(order, degree, polys):
    assert galois.primitive_poly(order, degree).coeffs.tolist() == polys[0]


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_primitive_poly_max(order, degree, polys):
    assert galois.primitive_poly(order, degree, method="max").coeffs.tolist() == polys[-1]


@pytest.mark.parametrize("order,degree,polys", PARAMS)
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


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_primitive_polys(order, degree, polys):
    assert [f.coeffs.tolist() for f in galois.primitive_polys(order, degree)] == polys


@pytest.mark.parametrize("order,degree,polys", PARAMS)
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


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_minimum_terms(order, degree, polys):
    min_terms = min(np.count_nonzero(f) for f in polys)
    min_term_polys = [f for f in polys if np.count_nonzero(f) == min_terms]
    assert [f.coeffs.tolist() for f in galois.primitive_polys(order, degree, terms="min")] == min_term_polys

    f = galois.primitive_poly(order, degree, terms="min", method="random")
    assert f.coeffs.tolist() in min_term_polys


@pytest.mark.timeout(1, method="signal")
@pytest.mark.parametrize("order,degree,polys", PARAMS_DB)
def test_minimum_terms_from_database(order, degree, polys):
    p = galois.primitive_poly(order, degree, terms="min")
    db_degrees = p.nonzero_degrees.tolist()
    db_coeffs = p.nonzero_coeffs.tolist()
    exp_degrees, exp_coeffs = polys
    assert db_degrees == exp_degrees and db_coeffs == exp_coeffs


def test_conway_poly_exceptions():
    with pytest.raises(TypeError):
        galois.conway_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.conway_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.conway_poly(4, 3)
    with pytest.raises(ValueError):
        galois.conway_poly(2, 0)
    with pytest.raises(LookupError):
        # GF(2^409) is the largest characteristic-2 field in Frank Luebeck's database
        galois.conway_poly(2, 410)


def test_conway_poly():
    assert galois.conway_poly(2, 8) == galois.Poly.Degrees([8, 4, 3, 2, 0])

    GF3 = galois.GF(3)
    assert galois.conway_poly(3, 8) == galois.Poly.Degrees([8, 5, 4, 2, 1, 0], coeffs=[1, 2, 1, 2, 2, 2], field=GF3)

    GF5 = galois.GF(5)
    assert galois.conway_poly(5, 8) == galois.Poly.Degrees([8, 4, 2, 1, 0], coeffs=[1, 1, 3, 4, 2], field=GF5)


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
