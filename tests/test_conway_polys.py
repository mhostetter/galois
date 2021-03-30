"""
A pytest module to test the Conway polynomial function.
"""
import numpy as np
import pytest

import galois


def test_valid():
    p = galois.conway_poly(2, 3)
    poly = galois.Poly([1,0,1,1])
    assert p == poly


def test_non_integer_characteristic():
    with pytest.raises(TypeError):
        p = galois.conway_poly(2.0, 3)


def test_non_integer_degree():
    with pytest.raises(TypeError):
        p = galois.conway_poly(2, 3.0)


def test_non_prime_characteristic():
    with pytest.raises(ValueError):
        p = galois.conway_poly(6, 3)


def test_non_positive_degree():
    with pytest.raises(ValueError):
        p = galois.conway_poly(2, 0)
    with pytest.raises(ValueError):
        p = galois.conway_poly(2, -2)


def test_not_found():
    # GF(2^409) is the largest 2-characteristic field in Frank Luebeck's database
    with pytest.raises(LookupError):
        p = galois.conway_poly(2, 410)
