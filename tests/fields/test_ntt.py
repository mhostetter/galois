"""
A pytest module to test the Number-Theoretic Transform (NTT) implementation.

Test vectors are generated using the `sympy` Python package and `sympy.ntt()` and `sympy.intt()` functions.
"""

import numpy as np
import pytest

import galois

# LUT[i] = (x, p, X), generated with `X = sympy.ntt(x, prime=p)`
NTT_LUTS = [
    ([1, 2, 3, 4], 5, [0, 4, 3, 2]),
    ([1, 2, 3, 4], 13, [10, 8, 11, 1]),
    ([1, 2, 3, 4], 17, [10, 6, 15, 7]),
    ([1, 2, 3, 4], 3 * 256 + 1, [10, 643, 767, 122]),
]


def test_ntt_exceptions():
    with pytest.raises(TypeError):
        galois.ntt(1)
    with pytest.raises(TypeError):
        galois.ntt([1, 2, 3, 4], size=6.0)
    with pytest.raises(TypeError):
        galois.ntt([1, 2, 3, 4], modulus=3 * 256 + 1.0)

    with pytest.raises(ValueError):
        GF = galois.GF(2**8)  # Invalid field for NTTs
        galois.ntt(GF([1, 2, 3, 4]))
    with pytest.raises(ValueError):
        galois.ntt([1, 2, 3, 4], size=3)
    with pytest.raises(ValueError):
        galois.ntt([1, 2, 3, 40], modulus=13)
    with pytest.raises(ValueError):
        galois.ntt([1, 2, 3, 40], modulus=43)
    with pytest.raises(ValueError):
        galois.ntt([1, 2, 3, 4], modulus=3 * 256 + 2)


@pytest.mark.parametrize(["x", "p", "X"], NTT_LUTS)
def test_ntt(x, p, X):
    GF = galois.GF(p)

    X_test = galois.ntt(tuple(x), modulus=p)
    assert isinstance(X_test, GF)
    assert np.array_equal(X_test, X)

    X_test = galois.ntt(list(x), modulus=p)
    assert isinstance(X_test, GF)
    assert np.array_equal(X_test, X)

    X_test = galois.ntt(np.array(x), modulus=p)
    assert isinstance(X_test, GF)
    assert np.array_equal(X_test, X)

    X_test = galois.ntt(GF(x))
    assert isinstance(X_test, GF)
    assert np.array_equal(X_test, X)


def test_ntt_zero_padding():
    x = [1, 2, 3, 4, 5, 6]
    X1 = galois.ntt(x, size=8)
    X2 = galois.ntt(x + [0] * 2)
    assert np.array_equal(X1, X2)

    x = [60, 50, 40, 30, 20, 10]
    X1 = galois.ntt(x, size=16)
    X2 = galois.ntt(x + [0] * 10)
    assert np.array_equal(X1, X2)


def test_intt_exceptions():
    with pytest.raises(TypeError):
        galois.intt(1)
    with pytest.raises(TypeError):
        galois.intt([10, 643, 767, 122], size=6.0)
    with pytest.raises(TypeError):
        galois.intt([10, 643, 767, 122], modulus=3 * 256 + 1.0)
    with pytest.raises(TypeError):
        galois.intt([10, 643, 767, 122], scaled=1)

    with pytest.raises(ValueError):
        GF = galois.GF(2**8)  # Invalid field for NTTs
        galois.intt(GF([1, 2, 3, 4]))
    with pytest.raises(ValueError):
        galois.intt([10, 643, 767, 122], size=3)
    with pytest.raises(ValueError):
        galois.intt([10, 643, 767, 122], modulus=13)


@pytest.mark.parametrize(["x", "p", "X"], NTT_LUTS)
def test_intt(x, p, X):
    GF = galois.GF(p)

    x_test = galois.intt(tuple(X), modulus=p)
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)

    x_test = galois.intt(list(X), modulus=p)
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)

    x_test = galois.intt(np.array(X), modulus=p)
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)

    x_test = galois.intt(GF(X))
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)


@pytest.mark.parametrize(["x", "p", "X"], NTT_LUTS)
def test_intt_unscaled(x, p, X):
    GF = galois.GF(p)
    N = len(x)
    x = GF(x) * GF(N)  # If X = NTT(x), then the unscaled INTT is INTT(X) = x*N

    x_test = galois.intt(tuple(X), modulus=p, scaled=False)
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)

    x_test = galois.intt(list(X), modulus=p, scaled=False)
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)

    x_test = galois.intt(np.array(X), modulus=p, scaled=False)
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)

    x_test = galois.intt(GF(X), scaled=False)
    assert isinstance(x_test, GF)
    assert np.array_equal(x_test, x)
