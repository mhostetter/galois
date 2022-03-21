"""
A pytest module to test common cyclic code functions.

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


def test_poly_to_generator_matrix_exceptions():
    with pytest.raises(TypeError):
        galois.poly_to_generator_matrix(15.0, GENERATOR_POLY)
    with pytest.raises(TypeError):
        galois.poly_to_generator_matrix(15.0, GENERATOR_POLY.coeffs)
    with pytest.raises(TypeError):
        galois.poly_to_generator_matrix(15, GENERATOR_POLY, systematic=1)


def test_poly_to_generator_matrix():
    G = galois.poly_to_generator_matrix(15, GENERATOR_POLY)
    assert np.array_equal(G, G_TRUTH)


def test_roots_to_parity_check_matrix_exceptions():
    GF = galois.GF(2**4)
    alpha = GF.primitive_element
    t = 3
    roots = alpha**np.arange(1, 2*t + 1)

    with pytest.raises(TypeError):
        galois.roots_to_parity_check_matrix(15.0, roots)
    with pytest.raises(TypeError):
        galois.roots_to_parity_check_matrix(15.0, roots.view(np.ndarray))


# TODO: Add tests for `roots_to_parity_check_matrix()`
