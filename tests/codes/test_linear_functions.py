"""
A pytest module to test common linear code functions.

Octave:
    [H, G] = cyclgen(15, 465)
    G = flipud(fliplr(G))
    H = flipud(fliplr(H))

References
----------
* https://octave.sourceforge.io/communications/function/cyclgen.html
"""
import pytest
import numpy as np

import galois

GENERATOR_POLY = galois.Poly.Int(465)

G_TRUTH = galois.GF2([
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
])

H_TRUTH = galois.GF2([
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
])


def test_generator_to_parity_check_matrix_exceptions():
    with pytest.raises(TypeError):
        galois.generator_to_parity_check_matrix(G_TRUTH.view(np.ndarray))


def test_generator_to_parity_check_matrix():
    H = galois.generator_to_parity_check_matrix(G_TRUTH)
    assert np.array_equal(H, H_TRUTH)


def test_parity_check_to_generator_matrix_exceptions():
    with pytest.raises(TypeError):
        galois.parity_check_to_generator_matrix(H_TRUTH.view(np.ndarray))


def test_parity_check_to_generator_matrix():
    G = galois.parity_check_to_generator_matrix(H_TRUTH)
    assert np.array_equal(G, G_TRUTH)
