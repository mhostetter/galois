"""
A pytest module to test type errors for polynomial arithmetic over Galois fields.
"""

import pytest

import galois


def test_true_division_exceptions():
    GF = galois.GF(7)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)

    with pytest.raises(NotImplementedError):
        f / g


def test_modular_power_exceptions():
    GF = galois.GF(7)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 20

    with pytest.raises(TypeError):
        pow(f.coeffs, power, g)
    with pytest.raises(TypeError):
        pow(f, float(power), g)
    with pytest.raises(ValueError):
        # TODO: Should this be a TypeError or ValueError?
        pow(f, power, g.coeffs)
    with pytest.raises(ValueError):
        pow(f, -power, g)
    with pytest.raises(TypeError):
        # TODO: Should this be a TypeError or ValueError?
        pow(f, -power, galois.Poly(g.coeffs, field=galois.GF(31)))
