"""
A pytest module to test the Galois field array class factory function :obj:`galois.GF`.
"""
import numpy as np
import pytest

import galois


def test_valid():
    GF = galois.GF(2)
    assert issubclass(GF, galois.GFArray)
    assert issubclass(type(GF), galois.GFMeta)

    GF = galois.GF(2**8)
    assert issubclass(GF, galois.GFArray)
    assert issubclass(type(GF), galois.GFMeta)

    GF = galois.GF(31)
    assert issubclass(GF, galois.GFArray)
    assert issubclass(type(GF), galois.GFMeta)


def test_non_integer_order():
    with pytest.raises(TypeError):
        GF = galois.GF(2.0**8)


# def test_irreducible_poly_invalid():
#     prim_poly = 285
#     with pytest.raises(TypeError):
#         GF = galois.GF(2**8, prim_poly=prim_poly)

#     prim_poly = 3  # x + 1
#     with pytest.raises(TypeError):
#         GF = galois.GF(2, prim_poly=prim_poly)


def test_non_bool_rebuild():
    with pytest.raises(TypeError):
        GF = galois.GF(2**8, rebuild=1)


def test_non_prime_characteristic():
    with pytest.raises(ValueError):
        GF = galois.GF(6**3)
