"""
A pytest module to test BCH decoding.

Test vectors generated from Octave with bchpoly().

References
----------
* https://octave.sourceforge.io/communications/function/bchpoly.html
"""
import random

import pytest
import numpy as np

import galois


def random_errors(N, n, max_errors):
    N_errors = np.random.randint(0, max_errors + 1, N)
    N_errors[0] = max_errors  # Ensure the max number of errors is present at least once
    E = galois.GF2.Zeros((N, n))
    for i in range(N):
        E[i, random.sample(list(range(n)), N_errors[i])] ^= 1
    return E, N_errors


def test_15_7_systematic_all_correctable():
    n, k = 15, 7
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


def test_15_7_systematic_some_uncorrectable():
    n, k = 15, 7
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


def test_15_7_non_systematic_all_correctable():
    n, k = 15, 7
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


def test_15_7_non_systematic_some_uncorrectable():
    n, k = 15, 7
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


def test_31_21_systematic_all_correctable():
    n, k = 31, 21
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


def test_31_21_systematic_some_uncorrectable():
    n, k = 31, 21
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


def test_31_21_non_systematic_all_correctable():
    n, k = 31, 21
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


def test_31_21_non_systematic_some_uncorrectable():
    n, k = 31, 21
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
