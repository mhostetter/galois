"""
A pytest module to test the DFT over arbitrary finite fields.
"""

import numpy as np
import pytest

import galois


def test_fft_exceptions():
    GF = galois.GF(2**8)
    x = GF([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        np.fft.fft(x, n=6)
    with pytest.raises(ValueError):
        np.fft.fft(x, axis=0)
    with pytest.raises(ValueError):
        np.fft.fft(x, norm="front")


def test_ifft_exceptions():
    GF = galois.GF(2**8)
    x = GF([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        np.fft.ifft(x, n=6)
    with pytest.raises(ValueError):
        np.fft.ifft(x, axis=0)
    with pytest.raises(ValueError):
        np.fft.ifft(x, norm="front")


@pytest.mark.parametrize("order", [31, 2**8, 3**5, 5**3, 2**100, 36893488147419103183])
def test_fft_poly(order):
    GF = galois.GF(order)

    n = _calculate_fft_length(GF.order)
    x = GF.Random(n)
    X1 = np.fft.fft(x)

    poly = galois.Poly(x, order="asc")
    omega = GF.primitive_root_of_unity(n)
    X2 = poly(omega ** np.arange(n))

    assert np.array_equal(X1, X2)


@pytest.mark.parametrize("order", [31, 2**8, 3**5, 5**3, 2**100, 36893488147419103183])
def test_fft_zero_pad(order):
    GF = galois.GF(order)

    n = _calculate_fft_length(GF.order)
    x = GF.Random(n)
    p = 3  # The non-zero terms
    x[p:] = 0
    X1 = np.fft.fft(x)
    X2 = np.fft.fft(x[0:p], n=n)

    assert np.array_equal(X1, X2)


@pytest.mark.parametrize("order", [31, 2**8, 3**5, 5**3, 2**100, 36893488147419103183])
def test_fft_ifft(order):
    GF = galois.GF(order)

    n = _calculate_fft_length(GF.order)
    x = GF.Random(n)
    X = np.fft.fft(x)
    xx = np.fft.ifft(X)

    assert np.array_equal(x, xx)


def _calculate_fft_length(order):
    # Over GF(q), an FFT of length n exists iff n divides (q - 1),
    # because GF(q)^× is cyclic of order (q - 1).
    #
    # We choose n by sampling a divisor of (q - 1) constructed from its prime factorization,
    # while keeping n reasonably small (≈ up to 1000) so the test stays fast and exercises
    # mixed-radix logic (radix-2/3/5/...) rather than trivial n=1 cases.
    max_n = 100
    group_order = order - 1

    if group_order <= 1:
        # If the multiplicative group is tiny, just test the largest possible n
        return 1

    primes, multiplicities = galois.factors(group_order)

    # Start from n = 1 and try to multiply in prime factors (with multiplicity)
    # without exceeding max_n. We randomize the "factor multiset" order so that
    # over repeated test runs we exercise different mixed-radix compositions.
    factor_multiset = []
    for p, e in zip(primes, multiplicities):
        factor_multiset.extend([p] * e)

    n = 1
    for p in factor_multiset:
        n *= p
        if n > min(max_n, group_order):
            n //= p
            break

    return n
