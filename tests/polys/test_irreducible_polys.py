"""
A pytest module to test generating irreducible polynomials over finite fields.
"""
import pytest

import galois

from .luts.irreducible_polys_2 import (
    IRREDUCIBLE_POLYS_2_1,
    IRREDUCIBLE_POLYS_2_2,
    IRREDUCIBLE_POLYS_2_3,
    IRREDUCIBLE_POLYS_2_4,
    IRREDUCIBLE_POLYS_2_5,
    IRREDUCIBLE_POLYS_2_6,
    IRREDUCIBLE_POLYS_2_7,
    IRREDUCIBLE_POLYS_2_8,
)
from .luts.irreducible_polys_3 import (
    IRREDUCIBLE_POLYS_3_1,
    IRREDUCIBLE_POLYS_3_2,
    IRREDUCIBLE_POLYS_3_3,
    IRREDUCIBLE_POLYS_3_4,
    IRREDUCIBLE_POLYS_3_5,
    IRREDUCIBLE_POLYS_3_6,
)
from .luts.irreducible_polys_4 import (
    IRREDUCIBLE_POLYS_4_1,
    IRREDUCIBLE_POLYS_4_2,
    IRREDUCIBLE_POLYS_4_3,
)
from .luts.irreducible_polys_5 import (
    IRREDUCIBLE_POLYS_5_1,
    IRREDUCIBLE_POLYS_5_2,
    IRREDUCIBLE_POLYS_5_3,
    IRREDUCIBLE_POLYS_5_4,
)
from .luts.irreducible_polys_9 import (
    IRREDUCIBLE_POLYS_9_1,
    IRREDUCIBLE_POLYS_9_2,
    IRREDUCIBLE_POLYS_9_3,
)
from .luts.irreducible_polys_25 import IRREDUCIBLE_POLYS_25_1, IRREDUCIBLE_POLYS_25_2

PARAMS = [
    (2, 1, IRREDUCIBLE_POLYS_2_1),
    (2, 2, IRREDUCIBLE_POLYS_2_2),
    (2, 3, IRREDUCIBLE_POLYS_2_3),
    (2, 4, IRREDUCIBLE_POLYS_2_4),
    (2, 5, IRREDUCIBLE_POLYS_2_5),
    (2, 6, IRREDUCIBLE_POLYS_2_6),
    (2, 7, IRREDUCIBLE_POLYS_2_7),
    (2, 8, IRREDUCIBLE_POLYS_2_8),
    (2**2, 1, IRREDUCIBLE_POLYS_4_1),
    (2**2, 2, IRREDUCIBLE_POLYS_4_2),
    (2**2, 3, IRREDUCIBLE_POLYS_4_3),
    (3, 1, IRREDUCIBLE_POLYS_3_1),
    (3, 2, IRREDUCIBLE_POLYS_3_2),
    (3, 3, IRREDUCIBLE_POLYS_3_3),
    (3, 4, IRREDUCIBLE_POLYS_3_4),
    (3, 5, IRREDUCIBLE_POLYS_3_5),
    (3, 6, IRREDUCIBLE_POLYS_3_6),
    (3**2, 1, IRREDUCIBLE_POLYS_9_1),
    (3**2, 2, IRREDUCIBLE_POLYS_9_2),
    (3**2, 3, IRREDUCIBLE_POLYS_9_3),
    (5, 1, IRREDUCIBLE_POLYS_5_1),
    (5, 2, IRREDUCIBLE_POLYS_5_2),
    (5, 3, IRREDUCIBLE_POLYS_5_3),
    (5, 4, IRREDUCIBLE_POLYS_5_4),
    (5**2, 1, IRREDUCIBLE_POLYS_25_1),
    (5**2, 2, IRREDUCIBLE_POLYS_25_2),
]


def test_irreducible_poly_exceptions():
    with pytest.raises(TypeError):
        galois.irreducible_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.irreducible_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2**2 * 3**2, 3)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 0)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 3, method="invalid-argument")


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_irreducible_poly_min(order, degree, polys):
    assert galois.irreducible_poly(order, degree).coeffs.tolist() == polys[0]


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_irreducible_poly_max(order, degree, polys):
    assert galois.irreducible_poly(order, degree, method="max").coeffs.tolist() == polys[-1]


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_irreducible_poly_random(order, degree, polys):
    assert galois.irreducible_poly(order, degree, method="random").coeffs.tolist() in polys


def test_irreducible_polys_exceptions():
    with pytest.raises(TypeError):
        next(galois.irreducible_polys(2.0, 3))
    with pytest.raises(TypeError):
        next(galois.irreducible_polys(2, 3.0))
    with pytest.raises(TypeError):
        next(galois.irreducible_polys(2, 3, reverse=1))
    with pytest.raises(ValueError):
        next(galois.irreducible_polys(2**2 * 3**2, 3))
    with pytest.raises(ValueError):
        next(galois.irreducible_polys(2, -1))


@pytest.mark.parametrize("order,degree,polys", PARAMS)
def test_irreducible_polys(order, degree, polys):
    assert [f.coeffs.tolist() for f in galois.irreducible_polys(order, degree)] == polys


def test_large_degree():
    """
    See https://github.com/mhostetter/galois/issues/360.
    """
    f = galois.Poly.Degrees([233, 74, 0])
    assert f.is_irreducible()
