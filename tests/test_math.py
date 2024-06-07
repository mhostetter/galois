"""
A pytest module to test the functions in _math.py.
"""

import numpy as np
import pytest

import galois


def test_gcd_exceptions():
    with pytest.raises(TypeError):
        galois.gcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.gcd(10, 12.0)


def test_gcd(egcd):
    X, Y, D = egcd["X"], egcd["Y"], egcd["D"]
    for x, y, d in zip(X, Y, D):
        assert galois.gcd(x, y) == d


def test_egcd_exceptions():
    with pytest.raises(TypeError):
        galois.egcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.egcd(10, 12.0)


def test_egcd(egcd):
    X, Y, D, S, T = egcd["X"], egcd["Y"], egcd["D"], egcd["S"], egcd["T"]
    for x, y, d, s, t in zip(X, Y, D, S, T):
        assert galois.egcd(x, y) == (d, s, t)


def test_lcm_exceptions():
    with pytest.raises(TypeError):
        galois.lcm(1.0, 2, 3)
    with pytest.raises(TypeError):
        galois.lcm(1, 2.0, 3)
    with pytest.raises(TypeError):
        galois.lcm(1, 2, 3.0)
    with pytest.raises(ValueError):
        galois.lcm()


def test_lcm(lcm):
    X, Z = lcm["X"], lcm["Z"]
    for x, z in zip(X, Z):
        assert galois.lcm(*x) == z


def test_prod_exceptions():
    with pytest.raises(TypeError):
        galois.prod(1, 2.0, 3)
    with pytest.raises(TypeError):
        galois.prod(1, 2, 3.0)
    with pytest.raises(ValueError):
        galois.prod()


def test_prod(prod):
    X, Z = prod["X"], prod["Z"]
    for x, z in zip(X, Z):
        assert galois.prod(*x) == z


def test_are_coprime_exceptions():
    with pytest.raises(TypeError):
        galois.are_coprime(3.0, 4, 5)
    with pytest.raises(TypeError):
        galois.are_coprime(3, 4.0, 5)
    with pytest.raises(TypeError):
        galois.are_coprime(3, 4, 5.0)
    with pytest.raises(ValueError):
        galois.are_coprime()


def test_are_coprime():
    assert galois.are_coprime(3, 4, 5)
    assert not galois.are_coprime(3, 5, 9, 11)
    assert not galois.are_coprime(2, 3, 7, 256)


def test_crt_exceptions():
    with pytest.raises(TypeError):
        galois.crt(np.array([0, 3, 4]), [3, 4, 5])
    with pytest.raises(TypeError):
        galois.crt([0, 3, 4], np.array([3, 4, 5]))
    with pytest.raises(TypeError):
        galois.crt([0, 3.0, 4], [3, 4, 5])
    with pytest.raises(TypeError):
        galois.crt([0, 3, 4], [3, 4.0, 5])
    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 5, 7])
    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 6])


def test_crt(crt):
    X, Y, Z = crt["X"], crt["Y"], crt["Z"]
    for x, y, z in zip(X, Y, Z):
        if z is not None:
            assert galois.crt(x, y) == z
        else:
            with pytest.raises(ValueError):
                galois.crt(x, y)


def test_isqrt_exceptions():
    with pytest.raises(TypeError):
        galois.isqrt(3.0)
    with pytest.raises(ValueError):
        galois.isqrt(-3)


def test_isqrt(isqrt):
    X, Z = isqrt["X"], isqrt["Z"]
    for x, z in zip(X, Z):
        assert galois.isqrt(x) == z


def test_iroot_exceptions():
    with pytest.raises(TypeError):
        galois.iroot(9.0, 3)
    with pytest.raises(TypeError):
        galois.iroot(9, 3.0)
    with pytest.raises(ValueError):
        galois.iroot(-9, 3)


def test_iroot(iroot):
    X, R, Z = iroot["X"], iroot["R"], iroot["Z"]
    for x, r, z in zip(X, R, Z):
        assert galois.iroot(x, r) == z

    assert galois.iroot(10, 1) == 10
    assert galois.iroot(0, 2) == 0


def test_ilog_exceptions():
    with pytest.raises(TypeError):
        galois.ilog(9.0, 2)
    with pytest.raises(TypeError):
        galois.ilog(9, 2.0)
    with pytest.raises(ValueError):
        galois.ilog(0, 2)
    with pytest.raises(ValueError):
        galois.ilog(-9, 2)
    with pytest.raises(ValueError):
        galois.ilog(9, 1)


def test_ilog(ilog):
    X, B, Z = ilog["X"], ilog["B"], ilog["Z"]
    for x, b, z in zip(X, B, Z):
        assert galois.ilog(x, b) == z

    assert galois.ilog(10, 10) == 1
