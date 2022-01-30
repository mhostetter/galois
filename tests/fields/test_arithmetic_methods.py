"""
A pytest module to test the accuracy of Galois field arithmetic methods/operations.
"""
import random

import pytest
import numpy as np

import galois


def test_additive_order(field_additive_order):
    GF, X, Z = field_additive_order["GF"], field_additive_order["X"], field_additive_order["Z"]
    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    z = x.additive_order()
    assert np.array_equal(z, Z)
    assert type(z) is np.ndarray


def test_multiplicative_order(field_multiplicative_order):
    GF, X, Z = field_multiplicative_order["GF"], field_multiplicative_order["X"], field_multiplicative_order["Z"]
    if GF.dtypes[-1] == np.object_:
        # FIXME: Skipping large fields because they're too slow
        return

    dtype = random.choice(GF.dtypes)
    x = X.astype(dtype)
    z = x.multiplicative_order()
    assert np.array_equal(z, Z)
    assert type(z) is np.ndarray

    with pytest.raises(ArithmeticError):
        GF(0).multiplicative_order()
    with pytest.raises(ArithmeticError):
        GF.Range(0, 2).multiplicative_order()
