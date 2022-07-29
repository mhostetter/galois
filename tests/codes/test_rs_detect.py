"""
A pytest module to test Reed-Solomon error detection.
"""
import pytest
import numpy as np

import galois

from .helper import random_errors, random_type

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
    with pytest.raises(ValueError):
        rs.detect(GF.Random(n + 1))

    # Non-systematic
    n, k = 15, 11
    rs = galois.ReedSolomon(n, k, systematic=False)
    GF = rs.field
    with pytest.raises(ValueError):
        rs.detect(GF.Random(n - 1))


class TestSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_no_errors(self, size):
        n, k = size[0], size[1]
        N = 100
        rs = galois.ReedSolomon(n, k)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        R = C

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected == False)

    @pytest.mark.parametrize("size", CODES)
    def test_all_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        rs = galois.ReedSolomon(n, k)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.d - 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= rs.d - 1))[0]

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)

    @pytest.mark.parametrize("size", CODES)
    def test_some_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        rs = galois.ReedSolomon(n, k)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.d + 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= rs.d - 1))[0]

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)


class TestSystematicShortened:
    @pytest.mark.parametrize("size", CODES)
    def test_no_errors(self, size):
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
        R = C

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected == False)

    @pytest.mark.parametrize("size", CODES)
    def test_all_detectable(self, size):
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
        E, N_errors = random_errors(GF, N, ns, rs.d - 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= rs.d - 1))[0]

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)

    @pytest.mark.parametrize("size", CODES)
    def test_some_detectable(self, size):
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
        E, N_errors = random_errors(GF, N, ns, rs.d + 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= rs.d - 1))[0]

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)


class TestNonSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_no_errors(self, size):
        n, k = size[0], size[1]
        N = 100
        rs = galois.ReedSolomon(n, k, systematic=False)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        R = C

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected == False)

    @pytest.mark.parametrize("size", CODES)
    def test_all_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        rs = galois.ReedSolomon(n, k, systematic=False)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.d - 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= rs.d - 1))[0]

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)

    @pytest.mark.parametrize("size", CODES)
    def test_some_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        rs = galois.ReedSolomon(n, k, systematic=False)
        GF = rs.field
        M = GF.Random((N, k))
        C = rs.encode(M)
        E, N_errors = random_errors(GF, N, n, rs.d + 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= rs.d - 1))[0]

        RR = random_type(R)
        detected = rs.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)
