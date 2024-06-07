"""
A pytest module to test the accuracy of FieldArray advanced arithmetic.
"""

import random

import numpy as np


def test_convolve(field_convolve):
    GF, X, Y, Z = field_convolve["GF"], field_convolve["X"], field_convolve["Y"], field_convolve["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        dtype = random.choice(GF.dtypes)
        x = x.astype(dtype)
        y = y.astype(dtype)

        z = np.convolve(x, y)
        assert np.array_equal(z, z_truth)
        assert type(z) is GF
        assert z.dtype == dtype
