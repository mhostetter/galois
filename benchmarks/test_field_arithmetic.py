"""
A pytest module to benchmark FieldArray arithmetic.
"""

import numpy as np
import pytest

import galois


class Base:
    # Placeholder variables
    order = 2
    ufunc_mode = "jit-calculate"
    N = -1

    def setup_method(self):
        self.GF = galois.GF(self.order, compile=self.ufunc_mode)

        np.random.seed(123456789)
        self.x = self.GF.Random(self.N)
        self.y = self.GF.Random(self.N, low=1)
        self.z = np.random.randint(0, 2 * self.GF.order, self.N)

    def test_add(self, benchmark):
        benchmark(np.add, self.x, self.y)

    def test_subtract(self, benchmark):
        benchmark(np.subtract, self.x, self.y)

    def test_multiply(self, benchmark):
        benchmark(np.multiply, self.x, self.y)

    def test_divide(self, benchmark):
        benchmark(np.divide, self.x, self.y)

    def test_power(self, benchmark):
        benchmark(np.power, self.x, self.z)

    def test_additive_inverse(self, benchmark):
        benchmark(np.negative, self.x)

    def test_multiplicative_inverse(self, benchmark):
        benchmark(np.reciprocal, self.y)

    def test_scalar_multiply(self, benchmark):
        benchmark(np.multiply, self.x, self.z)


@pytest.mark.benchmark(group="GF(2) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'")
class Test_GF2_calculate(Base):
    order = 2
    ufunc_mode = "jit-calculate"
    N = 100_000


@pytest.mark.benchmark(group="GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'")
class Test_GF2_8_lookup(Base):
    order = 2**8
    ufunc_mode = "jit-lookup"
    N = 100_000


@pytest.mark.benchmark(group="GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'")
class Test_GF2_8_calculate(Base):
    order = 2**8
    ufunc_mode = "jit-calculate"
    N = 100_000


@pytest.mark.benchmark(group="GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'")
class Test_GF257_lookup(Base):
    order = 257
    ufunc_mode = "jit-lookup"
    N = 100_000


@pytest.mark.benchmark(group="GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'")
class Test_GF257_calculate(Base):
    order = 257
    ufunc_mode = "jit-calculate"
    N = 100_000


@pytest.mark.benchmark(group="GF(3^5) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'")
class Test_GF3_5_lookup(Base):
    order = 3**5
    ufunc_mode = "jit-lookup"
    N = 100_000


@pytest.mark.benchmark(group="GF(3^5) Array Arithmetic: shape=(10_000,), ufunc_mode='jit-calculate'")
class Test_GF3_5_calculate(Base):
    order = 3**5
    ufunc_mode = "jit-calculate"
    N = 10_000
