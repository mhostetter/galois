"""
A pytest module to test Reed-Solomon codes.

Test vectors generated from Octave with rsgenpoly().

References
----------
* https://octave.sourceforge.io/communications/function/rsgenpoly.html
"""
import numpy as np

import galois


def test_rs_generator_poly():
    # S. Lin and D. Costello. Error Control Coding. Example 7.1, p. 238.
    p = galois.primitive_poly(2, 6, method="smallest")
    GF = galois.GF(2**6, irreducible_poly=p)
    a = GF.primitive_element
    assert galois.rs_generator_poly(63, 57) == galois.Poly(a**np.array([0, 59, 48, 43, 55, 10, 21]), field=GF)

    # Octave rsgenpoly()
    assert np.array_equal(galois.rs_generator_poly(15, 13).coeffs, [1, 6, 8])
    assert np.array_equal(galois.rs_generator_poly(15, 11).coeffs, [1, 13, 12, 8, 7])
    assert np.array_equal(galois.rs_generator_poly(15, 9).coeffs, [1, 7, 9, 3, 12, 10, 12])
    assert np.array_equal(galois.rs_generator_poly(15, 7).coeffs, [1, 9, 4, 3, 4, 13, 6, 14, 12])
    assert np.array_equal(galois.rs_generator_poly(15, 5).coeffs, [1, 4, 8, 10, 12, 9, 4, 2, 12, 2, 7])
    assert np.array_equal(galois.rs_generator_poly(15, 3).coeffs, [1, 5, 9, 5, 8, 1, 4, 13, 9, 4, 12, 13, 8])
    assert np.array_equal(galois.rs_generator_poly(15, 1).coeffs, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def test_rs_generator_poly_diff_c():
    # Octave rsgenpoly(15, k, 19, 2)
    assert np.array_equal(galois.rs_generator_poly(15, 13, c=2).coeffs, [1, 12, 6])
    assert np.array_equal(galois.rs_generator_poly(15, 11, c=2).coeffs, [1, 9, 5, 12, 9])

    # Octave rsgenpoly(15, k, 19, 3)
    assert np.array_equal(galois.rs_generator_poly(15, 13, c=3).coeffs, [1, 11, 11])
    assert np.array_equal(galois.rs_generator_poly(15, 11, c=3).coeffs, [1, 1, 7, 10, 8])

    # Octave rsgenpoly(15, k, 19, 4)
    assert np.array_equal(galois.rs_generator_poly(15, 13, c=4).coeffs, [1, 5, 10])
    assert np.array_equal(galois.rs_generator_poly(15, 11, c=4).coeffs, [1, 2, 15, 15, 11])


def test_rs_generator_poly_diff_primitive_poly():
    # Octave rsgenpoly(31, k, 41)
    p = galois.Poly.Degrees([5,3,0])
    assert np.array_equal(galois.rs_generator_poly(31, 29, primitive_poly=p).coeffs, [1, 6, 8])
    assert np.array_equal(galois.rs_generator_poly(31, 27, primitive_poly=p).coeffs, [1, 30, 7, 24, 19])

    # Octave rsgenpoly(31, k, 41, 2)
    p = galois.Poly.Degrees([5,3,0])
    assert np.array_equal(galois.rs_generator_poly(31, 29, primitive_poly=p, c=2).coeffs, [1, 12, 9])
    assert np.array_equal(galois.rs_generator_poly(31, 27, primitive_poly=p, c=2).coeffs, [1, 21, 28, 31, 3])

    # Octave rsgenpoly(31, k, 61)
    p = galois.Poly.Degrees([5,4,3,2,0])
    assert np.array_equal(galois.rs_generator_poly(31, 29, primitive_poly=p).coeffs, [1, 6, 8])
    assert np.array_equal(galois.rs_generator_poly(31, 27, primitive_poly=p).coeffs, [1, 30, 17, 16, 10])

    # Octave rsgenpoly(31, k, 61, 2)
    p = galois.Poly.Degrees([5,4,3,2,0])
    assert np.array_equal(galois.rs_generator_poly(31, 29, primitive_poly=p, c=2).coeffs, [1, 12, 29])
    assert np.array_equal(galois.rs_generator_poly(31, 27, primitive_poly=p, c=2).coeffs, [1, 1, 3, 14, 19])


def test_rs_parity_check_matrix():
    p = galois.Poly.Degrees([4,1,0])
    GF = galois.GF(2**4, irreducible_poly=p)
    alpha = GF.primitive_element
    H = galois.rs_parity_check_matrix(15, 11)
    H_truth = alpha**np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
    ])
    assert np.array_equal(H, np.fliplr(H_truth))  # NOTE: We use the convention of polynomial highest degree first, not last


# def test_rs_properties():
#     rs = galois.ReedSolomon(7, 4)
#     assert (rs.n, rs.k, rs.t) == (7, 4, 1)

#     rs = galois.ReedSolomon(15, 11)
#     assert (rs.n, rs.k, rs.t) == (15, 11, 1)

#     rs = galois.ReedSolomon(15, 7)
#     assert (rs.n, rs.k, rs.t) == (15, 7, 2)

#     rs = galois.ReedSolomon(15, 5)
#     assert (rs.n, rs.k, rs.t) == (15, 5, 3)
