"""
A pytest module to test BCH error detection.
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
        bch.detect(GF.Random(n + 1))

    # Non-systematic
    n, k = 15, 7
    bch = galois.BCH(n, k, systematic=False)
    GF = galois.GF2
    with pytest.raises(ValueError):
        bch.detect(GF.Random(n - 1))


class TestSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_no_errors(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        R = C

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected == False)

    @pytest.mark.parametrize("size", CODES)
    def test_all_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.d - 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= bch.d - 1))[0]

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)

    @pytest.mark.parametrize("size", CODES)
    def test_some_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.d + 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= bch.d - 1))[0]

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)


class TestSystematicShortened:
    @pytest.mark.parametrize("size", CODES)
    def test_no_errors(self, size):
        n, k = size[0], size[1]
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, ks))
        C = bch.encode(M)
        R = C

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected == False)

    @pytest.mark.parametrize("size", CODES)
    def test_all_detectable(self, size):
        n, k = size[0], size[1]
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, ks))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, ns, bch.d - 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= bch.d - 1))[0]

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)

    @pytest.mark.parametrize("size", CODES)
    def test_some_detectable(self, size):
        n, k = size[0], size[1]
        ks = k // 2  # Shorten the code in half
        ns = n - (k - ks)
        N = 100
        bch = galois.BCH(n, k)
        M = galois.GF2.Random((N, ks))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, ns, bch.d + 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= bch.d - 1))[0]

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)


class TestNonSystematic:
    @pytest.mark.parametrize("size", CODES)
    def test_no_errors(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k, systematic=False)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        R = C

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected == False)

    @pytest.mark.parametrize("size", CODES)
    def test_all_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k, systematic=False)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.d - 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= bch.d - 1))[0]

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)

    @pytest.mark.parametrize("size", CODES)
    def test_some_detectable(self, size):
        n, k = size[0], size[1]
        N = 100
        bch = galois.BCH(n, k, systematic=False)
        M = galois.GF2.Random((N, k))
        C = bch.encode(M)
        E, N_errors = random_errors(galois.GF2, N, n, bch.d + 1)
        R = C + E

        corr_idxs = np.where(N_errors == 0)[0]
        det_idxs = np.where(np.logical_and(0 < N_errors, N_errors <= bch.d - 1))[0]

        RR = random_type(R)
        detected = bch.detect(RR)
        assert type(detected) is np.ndarray
        assert np.all(detected[corr_idxs] == False)
        assert np.all(detected[det_idxs] == True)
