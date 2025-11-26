"""
A pytest module to benchmark FieldArray arithmetic.
"""

import itertools

import numpy as np
import pytest

import galois


@pytest.mark.parametrize("size_K", [1, 2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.benchmark()
def test_fft(size_K, benchmark):
    size = size_K * 256
    for order in itertools.count(size + 1, step=size):
        p, e = galois.factors(order)
        if len(p) == len(e) == 1:
            break
    GF = galois.GF(p[0], e[0])
    x = GF.Random(size)
    np.fft.fft(x)  # Don't benchmark the numba compilation
    benchmark(np.fft.fft, x)
