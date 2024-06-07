"""
A pytest module to test generating and testing Conway polynomials.
"""

import pytest

import galois


def test_conway_poly_exceptions():
    with pytest.raises(TypeError):
        galois.conway_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.conway_poly(2, 3.0)
    with pytest.raises(TypeError):
        galois.conway_poly(2, 3, search=1)

    with pytest.raises(ValueError):
        galois.conway_poly(4, 3)
    with pytest.raises(ValueError):
        galois.conway_poly(2, 0)
    with pytest.raises(LookupError):
        # GF(2^409) is the largest characteristic-2 field in Frank Luebeck's database
        galois.conway_poly(2, 410)


def test_conway_poly():
    f = galois.conway_poly(2, 8)
    assert f == "x^8 + x^4 + x^3 + x^2 + 1"
    assert f.field == galois.GF(2)

    f = galois.conway_poly(3, 8)
    assert f == "x^8 + 2x^5 + x^4 + 2x^2 + 2x + 2"
    assert f.field == galois.GF(3)

    f = galois.conway_poly(5, 8)
    assert f == "x^8 + x^4 + 3x^2 + 4x + 2"
    assert f.field == galois.GF(5)


def test_conway_poly_manual_search():
    characteristic = 7
    for degree in range(1, 4):
        f = galois._polys._conway._conway_poly_search(characteristic, degree)
        g = galois.conway_poly(characteristic, degree)
        assert f == g


def test_conway_poly_fallback_search():
    p = 70_001
    with pytest.raises(LookupError):
        galois.conway_poly(p, 1)

    f = galois.conway_poly(p, 1, search=True)
    assert f == "x + 69998"
    assert f.field == galois.GF(p)
