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

from .helper import random_type

CODES = [
    (15, 13, 1),  # GF(2^4) with t=1
    (15, 11, 2),  # GF(2^4) with t=2
    (15, 9, 1),  # GF(2^4) with t=3
    (15, 7, 2),  # GF(2^4) with t=4
    (15, 5, 1),  # GF(2^4) with t=5
    (15, 3, 2),  # GF(2^4) with t=6
    (15, 1, 1),  # GF(2^4) with t=7
    (16, 14, 2),  # GF(17) with t=1
    (16, 12, 1),  # GF(17) with t=2
    (16, 10, 2),  # GF(17) with t=3
    (26, 24, 1),  # GF(3^3) with t=1
    (26, 22, 2),  # GF(3^3) with t=2
    (26, 20, 3),  # GF(3^3) with t=3
]


def test_exceptions():
    # Systematic
    n, k = 15, 11
    rs = galois.ReedSolomon(n, k)
    GF = rs.field
    with pytest.raises(ValueError):
        rs.encode(GF.Random(k + 1))

    # Non-systematic
    n, k = 15, 11
    rs = galois.ReedSolomon(n, k, systematic=False)
    GF = rs.field
    with pytest.raises(ValueError):
        rs.encode(GF.Random(k - 1))


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

    mm = random_type(m)
    c = rs.encode(mm)
    assert type(c) is GF
    assert np.array_equal(c, c_truth)

    mm = random_type(m)
    c = rs.encode(mm, parity_only=True)
    assert type(c) is GF
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

    mm = random_type(m)
    c = rs.encode(mm)
    assert type(c) is GF
    assert np.array_equal(c, c_truth)

    with pytest.raises(ValueError):
        c = rs.encode(m, parity_only=True)


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

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_primitive_poly(self):
        """
        rsenc(gf(M, 4, 25), 15, 9, 'end')
        """
        p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="max")
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

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
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

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, self.k:])


class Test_n15_k9_shortened:
    n, k = 15, 9
    ns, ks = 15-4, 9-4
    M = np.array([
        [ 0,  6, 15, 10,  5],
        [ 0, 10,  0,  4, 15],
        [ 2, 12,  3,  3,  8],
        [13, 11,  0, 15,  3],
        [ 2, 12,  9,  6,  8],
        [10,  3,  8,  1,  7],
        [ 0,  3,  3,  8,  0],
        [ 5, 15,  5, 12, 13],
        [14,  1,  3,  4, 15],
        [14, 11, 15, 12,  8],
    ])

    def test_default(self):
        """
        NOTE: Octave produces the incorrect result for shortened RS codes (https://savannah.gnu.org/bugs/?func=detailitem&item_id=60800)
        Z = zeros(10, 4);
        C = rsenc(gf([Z,M], 4), 15, 9, 'end');
        C(:,5:end)
        """
        rs = galois.ReedSolomon(self.n, self.k)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [ 0,  6, 15, 10,  5,  4,  2,  1,  0,  9,  0],
            [ 0, 10,  0,  4, 15,  6, 14, 15, 10,  9,  2],
            [ 2, 12,  3,  3,  8,  6,  8,  2, 13,  9, 11],
            [13, 11,  0, 15,  3,  6,  4,  3, 15,  3, 12],
            [ 2, 12,  9,  6,  8,  7,  1,  8,  0,  9, 14],
            [10,  3,  8,  1,  7,  8,  2,  8,  1, 11,  7],
            [ 0,  3,  3,  8,  0,  8,  4,  0, 10,  2,  6],
            [ 5, 15,  5, 12, 13, 14, 15,  2, 10, 12, 14],
            [14,  1,  3,  4, 15,  0, 15, 15,  0, 14,  7],
            [14, 11, 15, 12,  8,  8, 14,  3,  8,  4, 14],
        ])

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])

    def test_diff_primitive_poly(self):
        """
        NOTE: Octave produces the incorrect result for shortened RS codes (https://savannah.gnu.org/bugs/?func=detailitem&item_id=60800)
        Z = zeros(10, 4);
        C = rsenc(gf([Z,M], 4, 25), 15, 9, 'end');
        C(:,5:end)
        """
        p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="max")
        rs = galois.ReedSolomon(self.n, self.k, primitive_poly=p)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [ 0,  6, 15, 10,  5,  9, 15,  2, 10, 14, 14],
            [ 0, 10,  0,  4, 15,  0,  4, 14,  0,  5,  7],
            [ 2, 12,  3,  3,  8,  4,  3,  1,  7,  5,  6],
            [13, 11,  0, 15,  3,  2,  1, 10,  1, 12,  8],
            [ 2, 12,  9,  6,  8,  8,  3,  2,  9,  8, 13],
            [10,  3,  8,  1,  7, 10,  9, 12, 15, 11,  2],
            [ 0,  3,  3,  8,  0, 11,  8,  5,  9, 14, 10],
            [ 5, 15,  5, 12, 13,  7,  9,  5,  2,  5, 11],
            [14,  1,  3,  4, 15, 11,  9, 12,  2, 11,  7],
            [14, 11, 15, 12,  8,  1,  1, 10,  8,  9,  6],
        ])

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])

    def test_diff_c(self):
        """
        NOTE: Octave produces the incorrect result for shortened RS codes (https://savannah.gnu.org/bugs/?func=detailitem&item_id=60800)
        Z = zeros(10, 4);
        C = rsenc(gf([Z,M], 4), 15, 9, 3, 1);
        C(:,5:end)
        """
        c = 3
        rs = galois.ReedSolomon(self.n, self.k, c=c)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [ 0,  6, 15, 10,  5, 14,  3,  9, 12,  4,  7],
            [ 0, 10,  0,  4, 15,  5,  4, 15, 13,  7,  6],
            [ 2, 12,  3,  3,  8,  3, 11, 12,  8, 14, 14],
            [13, 11,  0, 15,  3,  5,  6, 11,  5,  2,  4],
            [ 2, 12,  9,  6,  8, 11, 12, 11,  6,  9,  6],
            [10,  3,  8,  1,  7, 13, 14,  9, 15,  0,  3],
            [ 0,  3,  3,  8,  0, 11, 10,  4,  2,  2,  4],
            [ 5, 15,  5, 12, 13, 11,  9,  9,  7, 15,  5],
            [14,  1,  3,  4, 15,  9,  9, 10, 14,  6,  6],
            [14, 11, 15, 12,  8, 11,  3,  4,  9,  9,  7],
        ])

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])


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

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_primitive_poly(self):
        """
        rsenc(gf(M, 5, 61), 31, 23, 'end')
        """
        p = galois.Poly.Degrees([5, 4, 3, 2, 0])  # galois.primitive_poly(2, 5, method="max")
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

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
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

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, self.k:])


class Test_n31_k23_shortened:
    n, k = 31, 23
    ns, ks = 31-10, 23-10
    M = np.array([
        [ 0,  0,  7, 14, 23,  2,  9, 20, 10,  3, 25, 16, 30],
        [17, 17, 11, 18, 16,  9, 13, 10, 24, 24, 26, 12,  8],
        [29,  0,  0,  4, 22, 30, 20,  5, 23,  3, 16,  1,  4],
        [ 4, 24, 21, 28, 25,  9, 27,  9, 23,  0, 22, 20, 24],
        [26, 21,  4, 17, 15, 24, 28,  2, 28,  4, 19, 20, 26],
        [13, 20,  5,  6,  6, 16, 31, 30, 15,  7, 10, 29,  3],
        [31,  3, 18,  3, 15, 17, 19, 25, 27,  6, 12, 27, 28],
        [14, 17, 11, 15,  3,  3,  1, 11,  4,  7, 10, 18,  4],
        [ 9, 21, 17,  2,  3,  6, 24, 16, 19, 25, 10,  0, 30],
        [17,  6, 17, 21, 15, 10, 31,  8, 27, 27, 21,  9, 15],
    ])

    def test_default(self):
        """
        NOTE: Octave produces the incorrect result for shortened RS codes (https://savannah.gnu.org/bugs/?func=detailitem&item_id=60800)
        Z = zeros(10, 10);
        C = rsenc(gf([Z,M], 5), 31, 23, 'end');
        C(:,11:end)
        """
        rs = galois.ReedSolomon(self.n, self.k)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [ 0,  0,  7, 14, 23,  2,  9, 20, 10,  3, 25, 16, 30, 10,  8, 24, 30,  6,  9, 18,  8],
            [17, 17, 11, 18, 16,  9, 13, 10, 24, 24, 26, 12,  8, 16,  6, 16, 19,  4, 17,  5, 13],
            [29,  0,  0,  4, 22, 30, 20,  5, 23,  3, 16,  1,  4,  2,  8, 19, 27, 10, 28,  9, 22],
            [ 4, 24, 21, 28, 25,  9, 27,  9, 23,  0, 22, 20, 24,  0, 31, 22, 31, 10, 24, 14, 30],
            [26, 21,  4, 17, 15, 24, 28,  2, 28,  4, 19, 20, 26,  4, 20, 11, 28, 30, 10, 30, 15],
            [13, 20,  5,  6,  6, 16, 31, 30, 15,  7, 10, 29,  3, 14,  7, 27,  0, 10,  6, 23, 28],
            [31,  3, 18,  3, 15, 17, 19, 25, 27,  6, 12, 27, 28, 18, 27, 10, 31, 18,  4,  5, 15],
            [14, 17, 11, 15,  3,  3,  1, 11,  4,  7, 10, 18,  4,  1, 31, 21, 25, 17, 15, 11, 21],
            [ 9, 21, 17,  2,  3,  6, 24, 16, 19, 25, 10,  0, 30,  7, 19, 30, 27, 11,  8,  4,  7],
            [17,  6, 17, 21, 15, 10, 31,  8, 27, 27, 21,  9, 15, 12, 24, 30,  8,  7, 12, 11, 19],
       ])

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])

    def test_diff_primitive_poly(self):
        """
        NOTE: Octave produces the incorrect result for shortened RS codes (https://savannah.gnu.org/bugs/?func=detailitem&item_id=60800)
        Z = zeros(10, 10);
        C = rsenc(gf([Z,M], 5, 61), 31, 23, 'end');
        C(:,11:end)
        """
        p = galois.Poly.Degrees([5, 4, 3, 2, 0])  # galois.primitive_poly(2, 5, method="max")
        rs = galois.ReedSolomon(self.n, self.k, primitive_poly=p)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [ 0,  0,  7, 14, 23,  2,  9, 20, 10,  3, 25, 16, 30,  4,  3, 23, 12, 17, 20, 21,  4],
            [17, 17, 11, 18, 16,  9, 13, 10, 24, 24, 26, 12,  8,  9, 18, 31, 20, 19, 28, 17,  1],
            [29,  0,  0,  4, 22, 30, 20,  5, 23,  3, 16,  1,  4, 17, 25, 11,  7, 30, 24, 13, 25],
            [ 4, 24, 21, 28, 25,  9, 27,  9, 23,  0, 22, 20, 24, 25,  7, 29, 24,  2, 21, 10, 18],
            [26, 21,  4, 17, 15, 24, 28,  2, 28,  4, 19, 20, 26,  4, 30,  5,  5, 31, 18, 28,  7],
            [13, 20,  5,  6,  6, 16, 31, 30, 15,  7, 10, 29,  3, 20,  6,  3, 29, 17, 29,  7, 19],
            [31,  3, 18,  3, 15, 17, 19, 25, 27,  6, 12, 27, 28, 24,  4, 12, 25,  2, 15, 11, 17],
            [14, 17, 11, 15,  3,  3,  1, 11,  4,  7, 10, 18,  4, 31,  9,  6, 28, 12, 24, 24, 28],
            [ 9, 21, 17,  2,  3,  6, 24, 16, 19, 25, 10,  0, 30, 21,  6, 25, 11, 30, 19, 13,  0],
            [17,  6, 17, 21, 15, 10, 31,  8, 27, 27, 21,  9, 15, 18, 30, 17,  6, 17,  5, 20, 15],
        ])

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])

    def test_diff_c(self):
        """
        NOTE: Octave produces the incorrect result for shortened RS codes (https://savannah.gnu.org/bugs/?func=detailitem&item_id=60800)
        Z = zeros(10, 10);
        C = rsenc(gf([Z,M], 5), 31, 23, 3, 1);
        C(:,11:end)
        """
        c = 3
        rs = galois.ReedSolomon(self.n, self.k, c=c)
        GF = rs.field
        M = GF(self.M)
        C_truth = GF([
            [ 0,  0,  7, 14, 23,  2,  9, 20, 10,  3, 25, 16, 30, 17, 11, 25, 17,  4, 16,  8, 22],
            [17, 17, 11, 18, 16,  9, 13, 10, 24, 24, 26, 12,  8, 31,  4, 18,  8, 22,  9, 24, 15],
            [29,  0,  0,  4, 22, 30, 20,  5, 23,  3, 16,  1,  4, 12, 26,  8, 24,  9, 31, 27,  3],
            [ 4, 24, 21, 28, 25,  9, 27,  9, 23,  0, 22, 20, 24,  2,  2,  5,  2,  1, 22, 14, 26],
            [26, 21,  4, 17, 15, 24, 28,  2, 28,  4, 19, 20, 26, 25, 11, 31, 27, 14, 18, 15, 10],
            [13, 20,  5,  6,  6, 16, 31, 30, 15,  7, 10, 29,  3,  3,  5, 29, 13, 15,  4, 25,  1],
            [31,  3, 18,  3, 15, 17, 19, 25, 27,  6, 12, 27, 28, 13, 15, 23, 20,  2, 10, 21, 28],
            [14, 17, 11, 15,  3,  3,  1, 11,  4,  7, 10, 18,  4, 30, 29, 18,  3, 10, 25, 26, 20],
            [ 9, 21, 17,  2,  3,  6, 24, 16, 19, 25, 10,  0, 30,  7, 26, 15, 28, 29, 29,  6,  6],
            [17,  6, 17, 21, 15, 10, 31,  8, 27, 27, 21,  9, 15,  8, 22,  8, 24, 11, 24, 23, 21],
        ])

        MM = random_type(M)
        C = rs.encode(MM)
        assert type(C) is GF
        assert np.array_equal(C, C_truth)

        MM = random_type(M)
        C = rs.encode(MM, parity_only=True)
        assert type(C) is GF
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])
