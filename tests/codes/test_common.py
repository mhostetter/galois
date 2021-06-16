"""
A pytest module to test common cyclic code functions.

Test vectors generated from Octave with cyclgen().

References
----------
* https://octave.sourceforge.io/communications/function/cyclgen.html
"""
import pytest
import numpy as np

import galois


def test_bch_15_7():
    # [H, G] = cyclgen(15, 465)
    g = galois.Poly.Integer(465)
    G = galois.generator_poly_to_matrix(15, g)
    G_truth = np.array([
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],

    ])
    assert np.array_equal(G, np.flip(G_truth, axis=(0,1)))
