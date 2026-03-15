"""
A pytest module to test polynomial arithmetic methods over Galois fields.
"""

import numpy as np
import pytest

import galois


def test_reverse(poly_reverse):
    GF, X, Z = poly_reverse["GF"], poly_reverse["X"], poly_reverse["Z"]
    for x, z_truth in zip(X, Z):
        z = x.reverse()

        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_lift_exceptions():
    p = galois.Poly.Random(5)
    with pytest.raises(TypeError):
        p.lift(field=7)
    with pytest.raises(TypeError):
        p.lift(None)

    GF4 = galois.GF(2**2)
    p_ext = galois.Poly.Random(5, field=GF4)
    with pytest.raises(ValueError):
        p_ext.lift(galois.GF(2**4))

    GF5 = galois.GF(5)
    with pytest.raises(ValueError):
        p.lift(GF5)


def test_lift():
    GF = galois.GF(7)
    GF2 = galois.GF(7**2)
    p = galois.Poly([3, 0, 5, 2], field=GF)

    q = p.lift(GF2)
    assert q == galois.Poly([3, 0, 5, 2], field=GF2)
    assert q.field is GF2
    assert type(q.coeffs) is GF2

    same = p.lift(GF)
    assert same == p
    assert same.field is GF
    assert type(same.coeffs) is GF


def test_roots_exceptions():
    p = galois.Poly.Random(5)
    with pytest.raises(TypeError):
        p.roots(multiplicity=1)


def test_roots(poly_roots):
    GF, X, R, M = poly_roots["GF"], poly_roots["X"], poly_roots["R"], poly_roots["M"]

    # FIXME: Skip large fields because they're too slow
    if GF.order > 2**16:
        return

    for x, r_truth, m_truth in zip(X, R, M):
        r = x.roots()
        assert np.array_equal(r, r_truth)
        assert type(r) is GF

        r, m = x.roots(multiplicity=True)
        assert np.array_equal(r, r_truth)
        assert type(r) is GF
        assert np.array_equal(m, m_truth)
        assert type(m) is np.ndarray


def test_derivative_exceptions():
    p = galois.Poly.Random(5)
    with pytest.raises(TypeError):
        p.derivative(1.0)
    with pytest.raises(ValueError):
        p.derivative(0)
    with pytest.raises(ValueError):
        p.derivative(-1)


def test_derivative(poly_derivative):
    GF, X, Y, Z = poly_derivative["GF"], poly_derivative["X"], poly_derivative["Y"], poly_derivative["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        z = x.derivative(y)

        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF
