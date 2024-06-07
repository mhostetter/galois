"""
A pytest module to benchmark forward-error correction encoding/decoding.
"""

import numpy as np
import pytest

import galois


class Base:
    # Placeholder variables
    code_sys = galois.ReedSolomon(63, 55)
    code_non_sys = galois.ReedSolomon(63, 55, systematic=False)
    GF = code_sys.field
    N = 1_000

    def setup_method(self):
        np.random.seed(123456789)
        self.M = self.GF.Random((self.N, self.code_sys.k))  # Message array
        self.C_sys = self.code_sys.encode(self.M)
        self.C_non_sys = self.code_non_sys.encode(self.M)

        self.C_sys[:, 0 : self.code_sys.t] += self.GF.Random(low=1)  # Add t bit errors
        self.C_non_sys[:, 0 : self.code_non_sys.t] += self.GF.Random(low=1)  # Add t bit errors

    def test_encode_systematic(self, benchmark):
        benchmark(self.code_sys.encode, self.M)

    def test_encode_non_systematic(self, benchmark):
        benchmark(self.code_non_sys.encode, self.M)

    def test_decode_systematic(self, benchmark):
        benchmark(self.code_sys.decode, self.C_sys)

    def test_decode_non_systematic(self, benchmark):
        benchmark(self.code_non_sys.decode, self.C_non_sys)


@pytest.mark.benchmark(group="BCH(63, 39): N=1_000")
class TestBCH(Base):
    code_sys = galois.BCH(63, 39)
    code_non_sys = galois.BCH(63, 39, systematic=False)
    GF = galois.GF2
    N = 1_000


@pytest.mark.benchmark(group="RS(63, 55): N=1_000")
class TestReedSolomon(Base):
    code_sys = galois.ReedSolomon(63, 55)
    code_non_sys = galois.ReedSolomon(63, 55, systematic=False)
    GF = code_sys.field
    N = 1_000
