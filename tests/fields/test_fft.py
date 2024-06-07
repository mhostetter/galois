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

    n = 5  # The FFT size
    while (GF.order - 1) % n != 0:
        n += 1

    x = GF.Random(n)
    X1 = np.fft.fft(x)

    poly = galois.Poly(x, order="asc")
    omega = GF.primitive_root_of_unity(n)
    X2 = poly(omega ** np.arange(n))

    assert np.array_equal(X1, X2)


@pytest.mark.parametrize("order", [31, 2**8, 3**5, 5**3, 2**100, 36893488147419103183])
def test_fft_zero_pad(order):
    GF = galois.GF(order)

    n = 5  # The FFT size
    while (GF.order - 1) % n != 0:
        n += 1

    p = 3  # The non-zero terms
    x = GF.Random(n)
    x[p:] = 0
    X1 = np.fft.fft(x)
    X2 = np.fft.fft(x[0:p], n=n)
    assert np.array_equal(X1, X2)


@pytest.mark.parametrize("order", [31, 2**8, 3**5, 5**3, 2**100, 36893488147419103183])
def test_fft_ifft(order):
    GF = galois.GF(order)

    n = 5  # The FFT size
    while (GF.order - 1) % n != 0:
        n += 1

    x = GF.Random(n)
    X = np.fft.fft(x)
    xx = np.fft.ifft(X)
    assert np.array_equal(x, xx)
