"""
A pytest module to test BCH decoding.
"""
import random

import pytest
import numpy as np

import galois

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


def random_errors(N, n, max_errors):
    N_errors = np.random.randint(0, max_errors + 1, N)
    N_errors[0] = max_errors  # Ensure the max number of errors is present at least once
    E = galois.GF2.Zeros((N, n))
    for i in range(N):
        E[i, random.sample(list(range(n)), N_errors[i])] ^= 1
    return E, N_errors


class TestSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(N, n, bch.t)
        R = C + E

        DEC_M = bch.decode(R)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)

        DEC_M, N_corr = bch.decode(R, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

        DEC_M = bch.decode(R.view(np.ndarray))
        assert type(DEC_M) is np.ndarray
        assert np.array_equal(DEC_M, M)

        DEC_M, N_corr = bch.decode(R.view(np.ndarray), errors=True)
        assert type(DEC_M) is np.ndarray
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(N, n, bch.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= bch.t)[0]

        DEC_M = bch.decode(R)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        DEC_M, N_corr = bch.decode(R, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])

        DEC_M = bch.decode(R.view(np.ndarray))
        assert type(DEC_M) is np.ndarray
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        DEC_M, N_corr = bch.decode(R.view(np.ndarray), errors=True)
        assert type(DEC_M) is np.ndarray
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
        E, N_errors = random_errors(N, n, bch.t)
        R = C + E

        DEC_M = bch.decode(R)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)

        DEC_M, N_corr = bch.decode(R, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

        DEC_M = bch.decode(R.view(np.ndarray))
        assert type(DEC_M) is np.ndarray
        assert np.array_equal(DEC_M, M)

        DEC_M, N_corr = bch.decode(R.view(np.ndarray), errors=True)
        assert type(DEC_M) is np.ndarray
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k, systematic=False)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(N, n, bch.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= bch.t)[0]

        DEC_M = bch.decode(R)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        DEC_M, N_corr = bch.decode(R, errors=True)
        assert type(DEC_M) is galois.GF2
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])

        DEC_M = bch.decode(R.view(np.ndarray))
        assert type(DEC_M) is np.ndarray
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        DEC_M, N_corr = bch.decode(R.view(np.ndarray), errors=True)
        assert type(DEC_M) is np.ndarray
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])
