"""
A pytest module to test functions on Galois field polynomials.
"""
import pytest
import numpy as np

import galois


def test_poly_exp_mod():
    GF = galois.GF(31)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 20
    assert f**power % g == galois.poly_exp_mod(f, power, g)
