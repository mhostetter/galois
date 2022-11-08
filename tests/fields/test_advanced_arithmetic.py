"""
A pytest module to test the accuracy of FieldArray advanced arithmetic.
"""
import random

import pytest
import numpy as np

import galois


def test_convolve(field_convolve):
    GF, X, Y, Z = field_convolve["GF"], field_convolve["X"], field_convolve["Y"], field_convolve["Z"]
    for i in range(len(X)):
        dtype = random.choice(GF.dtypes)
        x = X[i].astype(dtype)
        y = Y[i].astype(dtype)

        z = np.convolve(x, y)
        assert np.array_equal(z, Z[i])
        assert type(z) is GF
        assert z.dtype == dtype
