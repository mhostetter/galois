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
