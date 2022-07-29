"""
A pytest module to test BCH decoding.
"""
import pytest
import numpy as np

import galois

from .helper import random_errors, random_type

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
        bch.decode(GF.Random(n + 1))

    # Non-systematic
    n, k = 15, 7
    bch = galois.BCH(n, k, systematic=False)
    GF = galois.GF2
    with pytest.raises(ValueError):
        bch.decode(GF.Random(n - 1))


class TestSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.t)
        R = C + E

        RR = random_type(R)
        DEC_M = bch.decode(RR)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)

        RR = random_type(R)
        DEC_M, N_corr = bch.decode(RR, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= bch.t)[0]

        RR = random_type(R)
        DEC_M = bch.decode(RR)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        RR = random_type(R)
        DEC_M, N_corr = bch.decode(RR, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])


class TestSystematicShortened:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k = size[0], size[1]
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, ks))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, ns, bch.t)
        R = C + E

        RR = random_type(R)
        DEC_M = bch.decode(RR)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)

        RR = random_type(R)
        DEC_M, N_corr = bch.decode(RR, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k = size[0], size[1]
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, ks))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, ns, bch.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= bch.t)[0]

        RR = random_type(R)
        DEC_M = bch.decode(RR)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        RR = random_type(R)
        DEC_M, N_corr = bch.decode(RR, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])


class TestNonSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k, systematic=False)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.t)
        R = C + E

        RR = random_type(R)
        DEC_M = bch.decode(RR)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)

        RR = random_type(R)
        DEC_M, N_corr = bch.decode(RR, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k, systematic=False)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= bch.t)[0]

        RR = random_type(R)
        DEC_M = bch.decode(RR)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        RR = random_type(R)
        DEC_M, N_corr = bch.decode(RR, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])
