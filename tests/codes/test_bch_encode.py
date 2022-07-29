"""
A pytest module to test BCH encoding.

Test vectors generated from Octave with bchenco().

References
----------
* https://octave.sourceforge.io/communications/function/bchenco.html
"""
import pytest
import numpy as np

import galois

from .helper import random_type

CODES = [
    (15, 11),  # GF(2^4) with t=1
    (15, 7),  # GF(2^4) with t=2
    (15, 5),  # GF(2^4) with t=3
    (31, 26),  # GF(2^5) with t=1
    (31, 21),  # GF(2^5) with t=2
    (31, 16),  # GF(2^5) with t=3
    (31, 11),  # GF(2^5) with t=5
    (31, 6),  # GF(2^5) with t=7
    (63, 57),  # GF(2^6) with t=1
    (63, 51),  # GF(2^6) with t=2
    (63, 45),  # GF(2^6) with t=3
    (63, 39),  # GF(2^6) with t=4
    (63, 36),  # GF(2^6) with t=5
    (63, 30),  # GF(2^6) with t=6
    (63, 24),  # GF(2^6) with t=7
]


def test_exceptions():
    # Systematic
    n, k = 15, 7
    bch = galois.BCH(n, k)
    GF = galois.GF2
    with pytest.raises(ValueError):
        bch.encode(GF.Random(k + 1))

    # Non-systematic
    n, k = 15, 7
    bch = galois.BCH(n, k, systematic=False)
    GF = galois.GF2
    with pytest.raises(ValueError):
        bch.encode(GF.Random(k - 1))


@pytest.mark.parametrize("size", CODES)
def test_systematic(size):
    n, k = size[0], size[1]
    bch = galois.BCH(n, k, systematic=True)
    m = galois.GF2.Random(k)
    c_truth = galois.GF2.Zeros(n)
    c_truth[0:k] = m
    r_poly = (galois.Poly(m) * galois.Poly.Degrees([n-k])) % bch.generator_poly
    c_truth[-r_poly.coeffs.size:] = -r_poly.coeffs

    mm = random_type(m)
    c = bch.encode(mm)
    assert type(c) is galois.GF2
    assert np.array_equal(c, c_truth)

    mm = random_type(m)
    c = bch.encode(mm, parity_only=True)
    assert type(c) is galois.GF2
    assert np.array_equal(c, c_truth[k:])


@pytest.mark.parametrize("size", CODES)
def test_non_systematic(size):
    n, k = size[0], size[1]
    bch = galois.BCH(n, k, systematic=False)
    m = galois.GF2.Random(k)
    c_poly = galois.Poly(m) * bch.generator_poly
    c_truth = galois.GF2.Zeros(n)
    c_truth[-c_poly.coeffs.size:] = c_poly.coeffs

    mm = random_type(m)
    c = bch.encode(mm)
    assert type(c) is galois.GF2
    assert np.array_equal(c, c_truth)

    with pytest.raises(ValueError):
        c = bch.encode(m, parity_only=True)


class Test_n15_k7:
    n, k = 15, 7
    M = galois.GF2([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0],
    ])

    def test_default(self):
        """
        g = bchpoly(15, 7)
        bchenco(M, 15, 7, g, 'end')
        """
        bch = galois.BCH(self.n, self.k)
        C_truth = galois.GF2([
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_primitive_poly(self):
        """
        g = bchpoly(15, 7, 25)
        bchenco(M, 15, 7, g, 'end')
        """
        p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="max")
        bch = galois.BCH(self.n, self.k, primitive_poly=p)
        C_truth = galois.GF2([
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, self.k:])


class Test_n15_k7_shortened:
    n, k = 15, 7
    ns, ks = 15-3, 7-3
    M = galois.GF2([
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 0],
    ])

    def test_default(self):
        """
        g = bchpoly(15, 7)
        bchenco(M, 15-3, 7-3, g, 'end')
        """
        bch = galois.BCH(self.n, self.k)
        C_truth = galois.GF2([
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])

    def test_diff_primitive_poly(self):
        """
        g = bchpoly(15, 7, 25)
        bchenco(M, 15-3, 7-3, g, 'end')
        """
        p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="max")
        bch = galois.BCH(self.n, self.k, primitive_poly=p)
        C_truth = galois.GF2([
            [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])


class Test_n31_k21:
    n, k = 31, 21
    M = galois.GF2([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    ])

    def test_default(self):
        """
        g = bchpoly(31, 21)
        bchenco(M, 31, 21, g, 'end')
        """
        bch = galois.BCH(self.n, self.k)
        C_truth = galois.GF2([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, self.k:])

    def test_diff_primitive_poly(self):
        """
        g = bchpoly(31, 21, 61)
        bchenco(M, 31, 21, g, 'end')
        """
        p = galois.Poly.Degrees([5, 4, 3, 2, 0])  # galois.primitive_poly(2, 5, method="max")
        bch = galois.BCH(self.n, self.k, primitive_poly=p)
        C_truth = galois.GF2([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, self.k:])


class Test_n31_k21_shortened:
    n, k = 31, 21
    ns, ks = 31-10, 21-10
    M = galois.GF2([
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    ])

    def test_default(self):
        """
        g = bchpoly(31, 21)
        bchenco(M, 31-10, 21-10, g, 'end')
        """
        bch = galois.BCH(self.n, self.k)
        C_truth = galois.GF2([
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])

    def test_diff_primitive_poly(self):
        """
        g = bchpoly(31, 21, 61)
        bchenco(M, 31-10, 21-10, g, 'end')
        """
        p = galois.Poly.Degrees([5, 4, 3, 2, 0])  # galois.primitive_poly(2, 5, method="max")
        bch = galois.BCH(self.n, self.k, primitive_poly=p)
        C_truth = galois.GF2([
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1],
        ])

        MM = random_type(self.M)
        C = bch.encode(MM)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth)

        MM = random_type(self.M)
        C = bch.encode(MM, parity_only=True)
        assert type(C) is galois.GF2
        assert np.array_equal(C, C_truth[:, -(self.n - self.k):])
