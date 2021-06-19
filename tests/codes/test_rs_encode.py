"""
A pytest module to test Reed-Solomon encoding.

Test vectors generated from Octave with rsenc().

References
----------
* https://octave.sourceforge.io/communications/function/rsenc.html
"""
import pytest
import numpy as np

import galois

CODES = [
    (15, 13),  # GF(2^4) with t=1
    (15, 11),  # GF(2^4) with t=2
    (15, 9),  # GF(2^4) with t=3
    (15, 7),  # GF(2^4) with t=4
    (15, 5),  # GF(2^4) with t=5
    (15, 3),  # GF(2^4) with t=6
    (15, 1),  # GF(2^4) with t=7
    (16, 14),  # GF(17) with t=1
    (16, 12),  # GF(17) with t=2
    (16, 10),  # GF(17) with t=3
    (26, 24),  # GF(3^3) with t=1
    (26, 22),  # GF(3^3) with t=2
    (26, 20),  # GF(3^3) with t=3
]


@pytest.mark.parametrize("size", CODES)
def test_systematic(size):
    n, k = size[0], size[1]
    rs = galois.ReedSolomon(n, k, systematic=True)
    GF = rs.field
    m = GF.Random(k)
    c_truth = GF.Zeros(n)
    c_truth[0:k] = m
    r_poly = (galois.Poly(m) * galois.Poly.Identity(GF)**(n-k)) % rs.generator_poly
    c_truth[-r_poly.coeffs.size:] = -r_poly.coeffs

    c = rs.encode(m)
    assert type(c) is GF
    assert np.array_equal(c, c_truth)

    c = rs.encode(m, parity_only=True)
    assert type(c) is GF
    assert np.array_equal(c, c_truth[k:])

    c = rs.encode(m.view(np.ndarray))
    assert type(c) is np.ndarray
    assert np.array_equal(c, c_truth)

    c = rs.encode(m.view(np.ndarray), parity_only=True)
    assert type(c) is np.ndarray
    assert np.array_equal(c, c_truth[k:])


@pytest.mark.parametrize("size", CODES)
def test_non_systematic(size):
    n, k = size[0], size[1]
    rs = galois.ReedSolomon(n, k, systematic=False)
    GF = rs.field
    m = GF.Random(k)
    c_poly = galois.Poly(m) * rs.generator_poly
    c_truth = GF.Zeros(n)
    c_truth[-c_poly.coeffs.size:] = c_poly.coeffs

    c = rs.encode(m)
    assert type(c) is GF
    assert np.array_equal(c, c_truth)

    with pytest.raises(ValueError):
        c = rs.encode(m, parity_only=True)

    c = rs.encode(m.view(np.ndarray))
    assert type(c) is np.ndarray
    assert np.array_equal(c, c_truth)

    with pytest.raises(ValueError):
        c = rs.encode(m.view(np.ndarray), parity_only=True)


def test_15_9():
    """
    rsenc(gf(M, 4), 15, 9, 'end')
    """
    n, k = 15, 9
    rs = galois.ReedSolomon(n, k)
    GF = rs.field
    M = GF([
        [ 5,  4, 13, 15,  6, 14,  8,  8, 10],
        [14, 10, 14,  0, 13,  2, 11, 13, 15],
        [ 8, 12,  7, 11,  4,  7, 12, 15,  7],
        [10, 15,  1, 14,  3,  8, 13, 14, 12],
        [12,  2, 15,  8,  3, 15,  6, 11,  4],
        [ 5, 10,  0, 14,  0, 14,  3, 14,  7],
        [ 0,  1,  1, 13, 14, 11, 12, 11,  3],
        [ 4,  0,  0, 13,  8, 10, 12, 10, 13],
        [ 2,  8, 10, 13,  8, 11,  5, 10,  5],
        [ 9,  3,  4,  7,  3,  7,  6, 11, 15],
    ])
    C_truth = GF([
        [ 5,  4, 13, 15,  6, 14,  8,  8, 10,  1,  7,  4,  2,  1,  4],
        [14, 10, 14,  0, 13,  2, 11, 13, 15,  5,  5,  7,  4,  5,  2],
        [ 8, 12,  7, 11,  4,  7, 12, 15,  7, 14,  5,  1, 12,  7, 14],
        [10, 15,  1, 14,  3,  8, 13, 14, 12, 13, 11,  5, 13,  6, 14],
        [12,  2, 15,  8,  3, 15,  6, 11,  4,  7, 10, 10, 12, 11, 11],
        [ 5, 10,  0, 14,  0, 14,  3, 14,  7, 15,  1,  1,  7, 14,  7],
        [ 0,  1,  1, 13, 14, 11, 12, 11,  3,  1,  5, 12,  7,  2,  9],
        [ 4,  0,  0, 13,  8, 10, 12, 10, 13,  7, 15,  7, 12, 11, 11],
        [ 2,  8, 10, 13,  8, 11,  5, 10,  5,  4, 12, 12,  1, 12, 10],
        [ 9,  3,  4,  7,  3,  7,  6, 11, 15, 13, 11,  7, 11,  3, 13],
    ])

    C = rs.encode(M)
    assert type(C) is GF
    assert np.array_equal(C, C_truth)

    C = rs.encode(M, parity_only=True)
    assert type(C) is GF
    assert np.array_equal(C, C_truth[:, k:])

    C = rs.encode(M.view(np.ndarray))
    assert type(C) is np.ndarray
    assert np.array_equal(C, C_truth)

    C = rs.encode(M.view(np.ndarray), parity_only=True)
    assert type(C) is np.ndarray
    assert np.array_equal(C, C_truth[:, k:])


def test_15_9_diff_primitive_poly():
    """
    rsenc(gf(M, 4, 25), 15, 9, 'end')
    """
    n, k = 15, 9
    p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="largest")
    rs = galois.ReedSolomon(n, k, primitive_poly=p)
    GF = rs.field
    M = GF([
        [ 5,  4, 13, 15,  6, 14,  8,  8, 10],
        [14, 10, 14,  0, 13,  2, 11, 13, 15],
        [ 8, 12,  7, 11,  4,  7, 12, 15,  7],
        [10, 15,  1, 14,  3,  8, 13, 14, 12],
        [12,  2, 15,  8,  3, 15,  6, 11,  4],
        [ 5, 10,  0, 14,  0, 14,  3, 14,  7],
        [ 0,  1,  1, 13, 14, 11, 12, 11,  3],
        [ 4,  0,  0, 13,  8, 10, 12, 10, 13],
        [ 2,  8, 10, 13,  8, 11,  5, 10,  5],
        [ 9,  3,  4,  7,  3,  7,  6, 11, 15],
    ])
    C_truth = GF([
        [ 5,  4, 13, 15,  6, 14,  8,  8, 10,  0,  6,  4,  1,  3,  7],
        [14, 10, 14,  0, 13,  2, 11, 13, 15,  2,  3,  2,  1,  0, 10],
        [ 8, 12,  7, 11,  4,  7, 12, 15,  7, 15,  8,  6,  8, 14, 12],
        [10, 15,  1, 14,  3,  8, 13, 14, 12, 14,  9, 14, 15,  7,  4],
        [12,  2, 15,  8,  3, 15,  6, 11,  4,  2,  1,  6,  0,  5,  4],
        [ 5, 10,  0, 14,  0, 14,  3, 14,  7, 14, 11,  8,  2, 15,  2],
        [ 0,  1,  1, 13, 14, 11, 12, 11,  3,  2,  3,  0,  7,  2,  6],
        [ 4,  0,  0, 13,  8, 10, 12, 10, 13,  8,  1,  9,  8, 11,  9],
        [ 2,  8, 10, 13,  8, 11,  5, 10,  5,  3, 15,  7, 13,  5,  0],
        [ 9,  3,  4,  7,  3,  7,  6, 11, 15,  4,  1,  7,  1,  1, 14],
    ])

    C = rs.encode(M)
    assert type(C) is GF
    assert np.array_equal(C, C_truth)

    C = rs.encode(M, parity_only=True)
    assert type(C) is GF
    assert np.array_equal(C, C_truth[:, k:])

    C = rs.encode(M.view(np.ndarray))
    assert type(C) is np.ndarray
    assert np.array_equal(C, C_truth)

    C = rs.encode(M.view(np.ndarray), parity_only=True)
    assert type(C) is np.ndarray
    assert np.array_equal(C, C_truth[:, k:])


def test_15_9_diff_c():
    """
    rsenc(gf(M, 4), 15, 9, 3, 1)
    """
    n, k = 15, 9
    c = 3
    rs = galois.ReedSolomon(n, k, c=c)
    GF = rs.field
    M = GF([
        [ 5,  4, 13, 15,  6, 14,  8,  8, 10],
        [14, 10, 14,  0, 13,  2, 11, 13, 15],
        [ 8, 12,  7, 11,  4,  7, 12, 15,  7],
        [10, 15,  1, 14,  3,  8, 13, 14, 12],
        [12,  2, 15,  8,  3, 15,  6, 11,  4],
        [ 5, 10,  0, 14,  0, 14,  3, 14,  7],
        [ 0,  1,  1, 13, 14, 11, 12, 11,  3],
        [ 4,  0,  0, 13,  8, 10, 12, 10, 13],
        [ 2,  8, 10, 13,  8, 11,  5, 10,  5],
        [ 9,  3,  4,  7,  3,  7,  6, 11, 15],
    ])
    C_truth = GF([
        [ 5,  4, 13, 15,  6, 14,  8,  8, 10, 13,  0, 13,  7,  9,  3],
        [14, 10, 14,  0, 13,  2, 11, 13, 15, 11,  3,  3, 15,  3, 14],
        [ 8, 12,  7, 11,  4,  7, 12, 15,  7,  2,  9,  3, 13, 13, 14],
        [10, 15,  1, 14,  3,  8, 13, 14, 12,  2,  6, 12, 15,  0, 13],
        [12,  2, 15,  8,  3, 15,  6, 11,  4, 12, 13,  2, 12, 13,  1],
        [ 5, 10,  0, 14,  0, 14,  3, 14,  7,  7,  4,  1, 13,  9,  5],
        [ 0,  1,  1, 13, 14, 11, 12, 11,  3,  0,  6,  9,  3, 13, 10],
        [ 4,  0,  0, 13,  8, 10, 12, 10, 13, 14,  0, 11,  8,  6, 14],
        [ 2,  8, 10, 13,  8, 11,  5, 10,  5,  6,  0, 12, 10,  9,  3],
        [ 9,  3,  4,  7,  3,  7,  6, 11, 15,  7, 14, 11,  8,  0, 12],
    ])

    C = rs.encode(M)
    assert type(C) is GF
    assert np.array_equal(C, C_truth)

    C = rs.encode(M, parity_only=True)
    assert type(C) is GF
    assert np.array_equal(C, C_truth[:, k:])

    C = rs.encode(M.view(np.ndarray))
    assert type(C) is np.ndarray
    assert np.array_equal(C, C_truth)

    C = rs.encode(M.view(np.ndarray), parity_only=True)
    assert type(C) is np.ndarray
    assert np.array_equal(C, C_truth[:, k:])
