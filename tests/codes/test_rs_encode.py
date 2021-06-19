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


class Test_n15_k9:
    n, k = 15, 9
    M = np.array([
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

    def test_default(self):
        """
        rsenc(gf(M, 4), 15, 9, 'end')
        """
        rs = galois.ReedSolomon(self.n, self.k)
        GF = rs.field
        M = GF(self.M)
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
        assert np.array_equal(C, C_truth[:, self.k:])

        C = rs.encode(M.view(np.ndarray))
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth)

        C = rs.encode(M.view(np.ndarray), parity_only=True)
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_primitive_poly(self):
        """
        rsenc(gf(M, 4, 25), 15, 9, 'end')
        """
        p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="largest")
        rs = galois.ReedSolomon(self.n, self.k, primitive_poly=p)
        GF = rs.field
        M = GF(self.M)
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
        assert np.array_equal(C, C_truth[:, self.k:])

        C = rs.encode(M.view(np.ndarray))
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth)

        C = rs.encode(M.view(np.ndarray), parity_only=True)
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_c(self):
        """
        rsenc(gf(M, 4), 15, 9, 3, 1)
        """
        c = 3
        rs = galois.ReedSolomon(self.n, self.k, c=c)
        GF = rs.field
        M = GF(self.M)
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
        assert np.array_equal(C, C_truth[:, self.k:])

        C = rs.encode(M.view(np.ndarray))
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth)

        C = rs.encode(M.view(np.ndarray), parity_only=True)
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth[:, self.k:])


class Test_n31_k23:
    n, k = 31, 23
    M = np.array([
        [26, 29, 13, 30, 28,  8,  8, 13, 12, 31, 15,  0, 18,  4, 13, 26,  7, 31, 28, 21, 31, 27, 12],
        [14, 30, 13,  1, 31,  1, 11,  5, 19, 27,  3,  7, 30, 22, 16, 27, 22, 14, 28, 25, 12, 27,  0],
        [16,  4,  7, 16, 24,  2, 18,  4, 27, 16,  8,  9, 20,  7,  2, 18,  2,  0,  6, 29, 19,  0,  9],
        [19, 25, 16,  0, 24, 30, 24, 29, 18, 21,  9, 30, 14, 14, 26, 19, 13,  6, 29, 15,  3, 22,  5],
        [ 2,  6,  4, 30,  6, 15, 21,  3, 13,  5, 15, 13, 15, 11, 27, 28, 30, 11, 24,  2, 22,  0,  2],
        [23, 27,  5, 15, 12,  0,  2,  1,  9, 11,  8,  6,  1, 19,  8, 26, 26, 29,  5, 27,  4, 30, 28],
        [11, 10, 21, 22, 22, 25, 26, 26, 10,  8,  4,  9, 30, 12, 13, 29,  9, 11, 11, 11, 21, 21, 17],
        [ 1, 23, 28,  6, 27, 17,  4, 20, 31,  4,  0,  3,  8,  7,  9, 26,  9,  1, 30, 22, 15,  3, 14],
        [12, 29, 20, 21,  2, 25,  9,  0, 12, 25,  2,  4, 11, 18, 26,  1, 20, 30,  9,  1, 28, 22,  3],
        [31, 24, 26, 21, 15, 30, 19, 14, 17,  7, 13, 20, 13,  7, 14,  1, 17, 20, 28, 13, 13, 30, 19],
    ])

    def test_default(self):
        """
        rsenc(gf(M, 5), 31, 23, 'end')
        """
        rs = galois.ReedSolomon(self.n, self.k)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [26, 29, 13, 30, 28,  8,  8, 13, 12, 31, 15,  0, 18,  4, 13, 26,  7, 31, 28, 21, 31, 27, 12, 18, 23, 26,  9,  8,  0, 18, 28],
            [14, 30, 13,  1, 31,  1, 11,  5, 19, 27,  3,  7, 30, 22, 16, 27, 22, 14, 28, 25, 12, 27,  0, 25, 25, 21, 12, 20, 22, 14, 24],
            [16,  4,  7, 16, 24,  2, 18,  4, 27, 16,  8,  9, 20,  7,  2, 18,  2,  0,  6, 29, 19,  0,  9, 18, 17, 31, 20, 27, 17,  3, 29],
            [19, 25, 16,  0, 24, 30, 24, 29, 18, 21,  9, 30, 14, 14, 26, 19, 13,  6, 29, 15,  3, 22,  5, 16,  9, 21,  1,  4,  3, 13, 21],
            [ 2,  6,  4, 30,  6, 15, 21,  3, 13,  5, 15, 13, 15, 11, 27, 28, 30, 11, 24,  2, 22,  0,  2,  2,  0, 15, 21, 22, 19, 25,  3],
            [23, 27,  5, 15, 12,  0,  2,  1,  9, 11,  8,  6,  1, 19,  8, 26, 26, 29,  5, 27,  4, 30, 28, 10,  1,  4, 14, 23, 16, 22,  5],
            [11, 10, 21, 22, 22, 25, 26, 26, 10,  8,  4,  9, 30, 12, 13, 29,  9, 11, 11, 11, 21, 21, 17, 31, 22,  2,  1, 31,  5,  0, 25],
            [ 1, 23, 28,  6, 27, 17,  4, 20, 31,  4,  0,  3,  8,  7,  9, 26,  9,  1, 30, 22, 15,  3, 14, 31,  4, 16, 20, 28, 19, 27,  0],
            [12, 29, 20, 21,  2, 25,  9,  0, 12, 25,  2,  4, 11, 18, 26,  1, 20, 30,  9,  1, 28, 22,  3,  5, 28, 14, 24, 28,  8, 21, 30],
            [31, 24, 26, 21, 15, 30, 19, 14, 17,  7, 13, 20, 13,  7, 14,  1, 17, 20, 28, 13, 13, 30, 19, 16,  2, 15,  2, 15,  6,  2,  2],
       ])

        C = rs.encode(M)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        C = rs.encode(M, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, self.k:])

        C = rs.encode(M.view(np.ndarray))
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth)

        C = rs.encode(M.view(np.ndarray), parity_only=True)
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_primitive_poly(self):
        """
        rsenc(gf(M, 5, 61), 31, 23, 'end')
        """
        p = galois.Poly.Degrees([5, 4, 3, 2, 0])  # galois.primitive_poly(2, 5, method="largest")
        rs = galois.ReedSolomon(self.n, self.k, primitive_poly=p)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [26, 29, 13, 30, 28,  8,  8, 13, 12, 31, 15,  0, 18,  4, 13, 26,  7, 31, 28, 21, 31, 27, 12,  8, 25, 22, 27, 30, 16, 12,  1],
            [14, 30, 13,  1, 31,  1, 11,  5, 19, 27,  3,  7, 30, 22, 16, 27, 22, 14, 28, 25, 12, 27,  0,  7, 14, 25, 30, 23,  5, 11, 15],
            [16,  4,  7, 16, 24,  2, 18,  4, 27, 16,  8,  9, 20,  7,  2, 18,  2,  0,  6, 29, 19,  0,  9,  6, 19, 25,  1, 18, 21, 26, 25],
            [19, 25, 16,  0, 24, 30, 24, 29, 18, 21,  9, 30, 14, 14, 26, 19, 13,  6, 29, 15,  3, 22,  5, 24, 30,  5, 17, 24, 20,  3, 15],
            [ 2,  6,  4, 30,  6, 15, 21,  3, 13,  5, 15, 13, 15, 11, 27, 28, 30, 11, 24,  2, 22,  0,  2,  2, 15,  2, 10, 25, 30, 18, 13],
            [23, 27,  5, 15, 12,  0,  2,  1,  9, 11,  8,  6,  1, 19,  8, 26, 26, 29,  5, 27,  4, 30, 28, 26,  9, 23,  3, 16,  4,  8, 31],
            [11, 10, 21, 22, 22, 25, 26, 26, 10,  8,  4,  9, 30, 12, 13, 29,  9, 11, 11, 11, 21, 21, 17, 18,  5,  7, 30, 18,  6, 10,  8],
            [ 1, 23, 28,  6, 27, 17,  4, 20, 31,  4,  0,  3,  8,  7,  9, 26,  9,  1, 30, 22, 15,  3, 14,  0,  7, 29, 27,  3, 19, 18,  3],
            [12, 29, 20, 21,  2, 25,  9,  0, 12, 25,  2,  4, 11, 18, 26,  1, 20, 30,  9,  1, 28, 22,  3, 20,  2, 30, 29, 18,  7, 14,  7],
            [31, 24, 26, 21, 15, 30, 19, 14, 17,  7, 13, 20, 13,  7, 14,  1, 17, 20, 28, 13, 13, 30, 19, 26, 31, 27,  7, 17, 26, 21, 16],
        ])

        C = rs.encode(M)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        C = rs.encode(M, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, self.k:])

        C = rs.encode(M.view(np.ndarray))
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth)

        C = rs.encode(M.view(np.ndarray), parity_only=True)
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_c(self):
        """
        rsenc(gf(M, 5), 31, 23, 3, 1)
        """
        c = 3
        rs = galois.ReedSolomon(self.n, self.k, c=c)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [26, 29, 13, 30, 28,  8,  8, 13, 12, 31, 15,  0, 18,  4, 13, 26,  7, 31, 28, 21, 31, 27, 12, 16, 24, 14, 26, 10,  1, 22, 26],
            [14, 30, 13,  1, 31,  1, 11,  5, 19, 27,  3,  7, 30, 22, 16, 27, 22, 14, 28, 25, 12, 27,  0, 28,  6, 20, 24, 27,  7, 21, 13],
            [16,  4,  7, 16, 24,  2, 18,  4, 27, 16,  8,  9, 20,  7,  2, 18,  2,  0,  6, 29, 19,  0,  9, 17, 18, 11,  8,  6,  1, 19, 19],
            [19, 25, 16,  0, 24, 30, 24, 29, 18, 21,  9, 30, 14, 14, 26, 19, 13,  6, 29, 15,  3, 22,  5,  7,  3, 29, 18, 13, 25, 16,  2],
            [ 2,  6,  4, 30,  6, 15, 21,  3, 13,  5, 15, 13, 15, 11, 27, 28, 30, 11, 24,  2, 22,  0,  2, 18, 31,  1,  2,  2, 16, 22, 19],
            [23, 27,  5, 15, 12,  0,  2,  1,  9, 11,  8,  6,  1, 19,  8, 26, 26, 29,  5, 27,  4, 30, 28, 18, 15, 15,  4, 15, 28,  3,  8],
            [11, 10, 21, 22, 22, 25, 26, 26, 10,  8,  4,  9, 30, 12, 13, 29,  9, 11, 11, 11, 21, 21, 17, 11, 11, 24,  2,  1, 14, 28, 26],
            [ 1, 23, 28,  6, 27, 17,  4, 20, 31,  4,  0,  3,  8,  7,  9, 26,  9,  1, 30, 22, 15,  3, 14, 12,  8, 17, 12, 29, 22, 16, 20],
            [12, 29, 20, 21,  2, 25,  9,  0, 12, 25,  2,  4, 11, 18, 26,  1, 20, 30,  9,  1, 28, 22,  3,  3,  3,  9, 17, 29, 14,  6,  9],
            [31, 24, 26, 21, 15, 30, 19, 14, 17,  7, 13, 20, 13,  7, 14,  1, 17, 20, 28, 13, 13, 30, 19, 27,  3, 23, 17,  5,  8,  4, 23],
        ])

        C = rs.encode(M)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        C = rs.encode(M, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, self.k:])

        C = rs.encode(M.view(np.ndarray))
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth)

        C = rs.encode(M.view(np.ndarray), parity_only=True)
        assert type(C) is np.ndarray
        assert np.array_equal(C, C_truth[:, self.k:])
