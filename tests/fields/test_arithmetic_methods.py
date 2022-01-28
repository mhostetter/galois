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
