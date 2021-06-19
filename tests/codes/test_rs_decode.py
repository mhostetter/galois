"""
A pytest module to test Reed-Solomon decoding.
"""
import random

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


def random_errors(GF, N, n, max_errors):
    N_errors = np.random.randint(0, max_errors + 1, N)
    N_errors[0] = max_errors  # Ensure the max number of errors is present at least once
    E = GF.Zeros((N, n))
    for i in range(N):
        E[i, random.sample(list(range(n)), N_errors[i])] = GF.Random(N_errors[i], low=1)
    return E, N_errors


@pytest.mark.parametrize("size", CODES)
def test_systematic_all_correctable(size):
    n, k = size[0], size[1]
    N = 100
    rs = galois.ReedSolomon(n, k)
    GF = rs.field
    M = GF.Random((N, k))
    C = rs.encode(M)
    E, N_errors = random_errors(GF, N, n, rs.t)
    R = C + E

    DEC_M = rs.decode(R)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M, M)

    DEC_M, N_corr = rs.decode(R, errors=True)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M, M)
    assert np.array_equal(N_corr, N_errors)

    DEC_M = rs.decode(R.view(np.ndarray))
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M, M)

    DEC_M, N_corr = rs.decode(R.view(np.ndarray), errors=True)
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M, M)
    assert np.array_equal(N_corr, N_errors)


@pytest.mark.parametrize("size", CODES)
def test_systematic_some_uncorrectable(size):
    n, k = size[0], size[1]
    N = 100
    rs = galois.ReedSolomon(n, k)
    GF = rs.field
    M = GF.Random((N, k))
    C = rs.encode(M)
    E, N_errors = random_errors(GF, N, n, rs.t + 1)
    R = C + E

    corr_idxs = np.where(N_errors <= rs.t)[0]

    DEC_M = rs.decode(R)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

    DEC_M, N_corr = rs.decode(R, errors=True)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
    assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])

    DEC_M = rs.decode(R.view(np.ndarray))
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

    DEC_M, N_corr = rs.decode(R.view(np.ndarray), errors=True)
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
    assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])


@pytest.mark.parametrize("size", CODES)
def test_non_systematic_all_correctable(size):
    n, k = size[0], size[1]
    N = 100
    rs = galois.ReedSolomon(n, k, systematic=False)
    GF = rs.field
    M = GF.Random((N, k))
    C = rs.encode(M)
    E, N_errors = random_errors(GF, N, n, rs.t)
    R = C + E

    DEC_M = rs.decode(R)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M, M)

    DEC_M, N_corr = rs.decode(R, errors=True)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M, M)
    assert np.array_equal(N_corr, N_errors)

    DEC_M = rs.decode(R.view(np.ndarray))
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M, M)

    DEC_M, N_corr = rs.decode(R.view(np.ndarray), errors=True)
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M, M)
    assert np.array_equal(N_corr, N_errors)


@pytest.mark.parametrize("size", CODES)
def test_non_systematic_some_uncorrectable(size):
    n, k = size[0], size[1]
    N = 100
    rs = galois.ReedSolomon(n, k, systematic=False)
    GF = rs.field
    M = GF.Random((N, k))
    C = rs.encode(M)
    E, N_errors = random_errors(GF, N, n, rs.t + 1)
    R = C + E

    corr_idxs = np.where(N_errors <= rs.t)[0]

    DEC_M = rs.decode(R)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

    DEC_M, N_corr = rs.decode(R, errors=True)
    assert type(DEC_M) is GF
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
    assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])

    DEC_M = rs.decode(R.view(np.ndarray))
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

    DEC_M, N_corr = rs.decode(R.view(np.ndarray), errors=True)
    assert type(DEC_M) is np.ndarray
    assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
    assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])
