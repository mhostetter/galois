"""
A pytest module to test Reed-Solomon decoding.
"""
import pytest
import numpy as np

import galois

from .helper import random_errors

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


def test_exceptions():
    # Systematic
    n, k = 15, 11
    rs = galois.ReedSolomon(n, k)
    GF = rs.field
    with pytest.raises(TypeError):
        rs.decode(GF.Random(n).tolist())
    with pytest.raises(ValueError):
        rs.decode(GF.Random(n + 1))

    # Non-systematic
    n, k = 15, 11
    rs = galois.ReedSolomon(n, k, systematic=False)
    GF = rs.field
    with pytest.raises(TypeError):
        rs.decode(GF.Random(n).tolist())
    with pytest.raises(ValueError):
        rs.decode(GF.Random(n - 1))


class TestSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
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
    def test_some_uncorrectable(self, size):
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


class TestSystematicShortened:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k = size[0], size[1]
        if k == 1:
            return
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        rs = galois.ReedSolomon(n, k)
        GF = rs.field
        M = GF.Random((N, ks))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, ns, rs.t)
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
    def test_some_uncorrectable(self, size):
        n, k = size[0], size[1]
        if k == 1:
            return
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        rs = galois.ReedSolomon(n, k)
        GF = rs.field
        M = GF.Random((N, ks))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, ns, rs.t + 1)
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


class TestNonSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
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
    def test_some_uncorrectable(self, size):
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
