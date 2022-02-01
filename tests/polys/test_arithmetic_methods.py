"""
A pytest module to test polynomial arithmetic methods over Galois fields.
"""
import pytest
import numpy as np

import galois


def test_reverse(poly_reverse):
    GF, X, Z = poly_reverse["GF"], poly_reverse["X"], poly_reverse["Z"]
    for i in range(len(X)):
        x = X[i]
        z = x.reverse()

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_roots_exceptions():
    p = galois.Poly.Random(5)
    with pytest.raises(TypeError):
        p.roots(multiplicity=1)


def test_roots(poly_roots):
    GF, X, R, M = poly_roots["GF"], poly_roots["X"], poly_roots["R"], poly_roots["M"]

    # FIXME: Skip large fields because they're too slow
    if GF.order > 2**16:
        return

    for i in range(len(X)):
        x = X[i]

        r = x.roots()
        assert np.array_equal(r, R[i])
        assert type(r) is GF

        r, m = x.roots(multiplicity=True)
        assert np.array_equal(r, R[i])
        assert type(r) is GF
        assert np.array_equal(m, M[i])
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
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        z = x.derivative(y)

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF
