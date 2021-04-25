"""
A pytest module to test Galois field polynomial arithmetic.
"""
import numpy as np
import pytest

import galois


def degrees(x):
    return np.arange(x.coeffs.size - 1, -1, -1)


def convert_poly(x, poly_type):
    if poly_type == "dense":
        pass
    elif poly_type == "sparse":
        x = galois.poly.SparsePoly(degrees(x), x.coeffs, field=x.field)
    else:
        raise AssertionError

    return x


def convert_polys(x, y, poly_types):
    if poly_types == "dense-dense":
        pass
    elif poly_types == "dense-sparse":
        y = galois.poly.SparsePoly(degrees(y), y.coeffs, field=y.field)
    elif poly_types == "sparse-dense":
        x = galois.poly.SparsePoly(degrees(x), x.coeffs, field=x.field)
    elif poly_types == "sparse-sparse":
        x = galois.poly.SparsePoly(degrees(x), x.coeffs, field=x.field)
        y = galois.poly.SparsePoly(degrees(y), y.coeffs, field=y.field)
    else:
        raise AssertionError

    return x, y


@pytest.mark.parametrize("poly_types", ["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"])
def test_add(poly_add, poly_types):
    GF, X, Y, Z = poly_add["GF"], poly_add["X"], poly_add["Y"], poly_add["Z"]
    for i in range(len(X)):
        x, y = convert_polys(X[i], Y[i], poly_types)
        z = x + y

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


@pytest.mark.parametrize("poly_types", ["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"])
def test_subtract(poly_subtract, poly_types):
    GF, X, Y, Z = poly_subtract["GF"], poly_subtract["X"], poly_subtract["Y"], poly_subtract["Z"]
    for i in range(len(X)):
        x, y = convert_polys(X[i], Y[i], poly_types)
        z = x - y

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


@pytest.mark.parametrize("poly_types", ["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"])
def test_multiply(poly_multiply, poly_types):
    GF, X, Y, Z = poly_multiply["GF"], poly_multiply["X"], poly_multiply["Y"], poly_multiply["Z"]
    for i in range(len(X)):
        x, y = convert_polys(X[i], Y[i], poly_types)
        z = x * y

        assert z == Z[i]
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


@pytest.mark.parametrize("poly_types", ["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"])
def test_divmod(poly_divmod, poly_types):
    GF, X, Y, Q, R = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["Q"], poly_divmod["R"]
    for i in range(len(X)):
        x, y = convert_polys(X[i], Y[i], poly_types)
        q, r = divmod(x, y)

        assert q == Q[i]
        assert isinstance(q, galois.Poly)
        assert q.field is GF
        assert type(q.coeffs) is GF

        assert r == R[i]
        assert isinstance(r, galois.Poly)
        assert r.field is GF
        assert type(r.coeffs) is GF


@pytest.mark.parametrize("poly_types", ["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"])
def test_mod(poly_divmod, poly_types):
    # NOTE: Test modulo separately because there's a separate method to compute it without the quotient for space spacings
    GF, X, Y, R = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["R"]
    for i in range(len(X)):
        x, y = convert_polys(X[i], Y[i], poly_types)
        r = x % y
        assert r == R[i]
        assert isinstance(r, galois.Poly)
        assert r.field is GF
        assert type(r.coeffs) is GF


@pytest.mark.parametrize("poly_type", ["dense", "sparse"])
def test_power(poly_power, poly_type):
    GF, X, Y, Z = poly_power["GF"], poly_power["X"], poly_power["Y"], poly_power["Z"]
    for i in range(len(X)):
        x = convert_poly(X[i], poly_type)
        for j in range(len(Y)):
            y = Y[j]  # Integer exponent
            z = x ** y
            assert z == Z[i][j]
            assert isinstance(z, galois.Poly)
            assert z.field is GF
            assert type(z.coeffs) is GF


# @pytest.mark.parametrize("poly_type", ["dense", "sparse"])
@pytest.mark.parametrize("poly_type", ["dense"])
def test_evaluate_constant(poly_evaluate, poly_type):
    GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
    for i in range(len(X)):
        for j in range(Y.size):
            x = convert_poly(X[i], poly_type)  # Polynomial
            y = Y[j]  # GF element
            z = x(y)  # GF element
            assert z == Z[i,j]
            assert type(z) is GF


# @pytest.mark.parametrize("poly_type", ["dense", "sparse"])
@pytest.mark.parametrize("poly_type", ["dense"])
def test_evaluate_vector(poly_evaluate, poly_type):
    GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
    for i in range(len(X)):
        x = convert_poly(X[i], poly_type)  # Polynomial
        y = Y  # GF array
        z = x(y)  # GF array
        assert np.all(z == Z[i,:])
        assert type(z) is GF
