"""
A pytest module to test polynomial arithmetic over Galois fields.
"""
import pytest
import numpy as np

import galois


def test_add(poly_add):
    GF, X, Y, Z = poly_add["GF"], poly_add["X"], poly_add["Y"], poly_add["Z"]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        z = x + y

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_subtract(poly_subtract):
    GF, X, Y, Z = poly_subtract["GF"], poly_subtract["X"], poly_subtract["Y"], poly_subtract["Z"]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        z = x - y

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_multiply(poly_multiply):
    GF, X, Y, Z = poly_multiply["GF"], poly_multiply["X"], poly_multiply["Y"], poly_multiply["Z"]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        z = x * y

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_scalar_multiply(poly_scalar_multiply):
    GF, X, Y, Z = poly_scalar_multiply["GF"], poly_scalar_multiply["X"], poly_scalar_multiply["Y"], poly_scalar_multiply["Z"]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        z = x * y
        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF

        z = y * x
        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_divmod(poly_divmod):
    GF, X, Y, Q, R = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["Q"], poly_divmod["R"]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        q, r = divmod(x, y)

        assert q == Q[i]
        assert isinstance(q, galois.Poly)
        assert q.field is GF
        assert type(q.coeffs) is GF

        assert r == R[i]
        assert isinstance(r, galois.Poly)
        assert r.field is GF
        assert type(r.coeffs) is GF


def test_mod(poly_divmod):
    # NOTE: Test modulo separately because there's a separate method to compute it without the quotient for space spacings
    GF, X, Y, R = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["R"]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        r = x % y
        assert r == R[i]
        assert isinstance(r, galois.Poly)
        assert r.field is GF
        assert type(r.coeffs) is GF


def test_power(poly_power):
    GF, X, Y, Z = poly_power["GF"], poly_power["X"], poly_power["Y"], poly_power["Z"]
    for i in range(len(X)):
        x = X[i]  # Polynomial
        for j in range(len(Y)):
            y = Y[j]  # Integer exponent
            z = x ** y
            assert z == Z[i][j]
            assert isinstance(z, galois.Poly)
            assert z.field is GF
            assert type(z.coeffs) is GF


def test_evaluate_constant(poly_evaluate):
    GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
    for i in range(len(X)):
        j = np.random.default_rng().integers(0, Y.size)
        x = X[i]  # Polynomial
        y = Y[j]  # GF element
        z = x(y)  # GF element
        assert z == Z[i,j]
        assert type(z) is GF


def test_evaluate_vector(poly_evaluate):
    GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
    for i in range(len(X)):
        x = X[i]  # Polynomial
        y = Y  # GF array
        z = x(y)  # GF array
        assert np.all(z == Z[i,:])
        assert type(z) is GF
