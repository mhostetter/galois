"""
A pytest module to test the Galois field array class factory function :obj:`galois.GF_factory`.
"""
import numpy as np
import pytest

import galois


def test_valid():
    GF = galois.GF_factory(2**8)
    assert issubclass(GF, galois.GF)
    assert issubclass(GF, galois.gf2m.GF2m)

    GF = galois.GF_factory(31)
    assert issubclass(GF, galois.GF)
    assert issubclass(GF, galois.gfp.GFp)


def test_non_integer_order():
    with pytest.raises(TypeError):
        GF = galois.GF_factory(2.0**8)


def test_prim_poly_invalid():
    prim_poly = 285
    with pytest.raises(TypeError):
        GF = galois.GF_factory(2**8, prim_poly=prim_poly)

    prim_poly = 3  # x + 1
    with pytest.raises(TypeError):
        GF = galois.GF_factory(2, prim_poly=prim_poly)


def test_non_bool_rebuild():
    with pytest.raises(TypeError):
        GF = galois.GF_factory(2**8, rebuild=1)


def test_non_prime_characteristic():
    with pytest.raises(ValueError):
        GF = galois.GF_factory(6**3)
