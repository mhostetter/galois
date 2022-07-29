"""
A pytest module to test Reed-Solomon decoding.
"""
import pytest
import numpy as np

import galois

from .helper import random_errors, random_type

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
        rs.decode(GF.Random(n + 1))

    # Non-systematic
    n, k = 15, 11
    rs = galois.ReedSolomon(n, k, systematic=False)
    GF = rs.field
    with pytest.raises(ValueError):
        rs.decode(GF.Random(n - 1))


class TestSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k, c = size[0], size[1], size[2]
        N = 100
        rs = galois.ReedSolomon(n, k, c=c)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.t)
        R = C + E

        RR = random_type(R)
        DEC_M = rs.decode(RR)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M, M)

        RR = random_type(R)
        DEC_M, N_corr = rs.decode(RR, errors=True)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k, c = size[0], size[1], size[2]
        N = 100
        rs = galois.ReedSolomon(n, k, c=c)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= rs.t)[0]

        RR = random_type(R)
        DEC_M = rs.decode(RR)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        RR = random_type(R)
        DEC_M, N_corr = rs.decode(RR, errors=True)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])


class TestSystematicShortened:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k, c = size[0], size[1], size[2]
        if k == 1:
            return
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        rs = galois.ReedSolomon(n, k, c=c)
        GF = rs.field
        M = GF.Random((N, ks))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, ns, rs.t)
        R = C + E

        RR = random_type(R)
        DEC_M = rs.decode(RR)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M, M)

        RR = random_type(R)
        DEC_M, N_corr = rs.decode(RR, errors=True)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k, c = size[0], size[1], size[2]
        if k == 1:
            return
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        rs = galois.ReedSolomon(n, k, c=c)
        GF = rs.field
        M = GF.Random((N, ks))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, ns, rs.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= rs.t)[0]

        RR = random_type(R)
        DEC_M = rs.decode(RR)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        RR = random_type(R)
        DEC_M, N_corr = rs.decode(RR, errors=True)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])


class TestNonSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_all_correctable(self, size):
        n, k, c = size[0], size[1], size[2]
        N = 100
        rs = galois.ReedSolomon(n, k, c=c, systematic=False)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.t)
        R = C + E

        RR = random_type(R)
        DEC_M = rs.decode(RR)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M, M)

        RR = random_type(R)
        DEC_M, N_corr = rs.decode(RR, errors=True)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M, M)
        assert np.array_equal(N_corr, N_errors)

    @pytest.mark.parametrize("size", CODES)
    def test_some_uncorrectable(self, size):
        n, k, c = size[0], size[1], size[2]
        N = 100
        rs = galois.ReedSolomon(n, k, c=c, systematic=False)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.t + 1)
        R = C + E

        corr_idxs = np.where(N_errors <= rs.t)[0]

        RR = random_type(R)
        DEC_M = rs.decode(RR)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

        RR = random_type(R)
        DEC_M, N_corr = rs.decode(RR, errors=True)
        assert type(DEC_M) is GF
        assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
        assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])
