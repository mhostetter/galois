"""
A pytest module to test the functions in _math.py.
"""
import pytest

import galois


def test_gcd_exceptions():
    with pytest.raises(TypeError):
        galois.gcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.gcd(10, 12.0)


def test_gcd(egcd):
    X, Y, D = egcd["X"], egcd["Y"], egcd["D"]
    for i in range(len(X)):
        assert galois.gcd(X[i], Y[i]) == D[i]


def test_egcd_exceptions():
    with pytest.raises(TypeError):
        galois.egcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.egcd(10, 12.0)


def test_egcd(egcd):
    X, Y, D, S, T = egcd["X"], egcd["Y"], egcd["D"], egcd["S"], egcd["T"]
    for i in range(len(X)):
        assert galois.egcd(X[i], Y[i]) == (D[i], S[i], T[i])


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
    for i in range(len(X)):
        assert galois.lcm(*X[i]) == Z[i]


def test_prod_exceptions():
    with pytest.raises(TypeError):
        galois.prod(1, 2.0, 3)
    with pytest.raises(TypeError):
        galois.prod(1, 2, 3.0)
    with pytest.raises(ValueError):
        galois.prod()


def test_prod(prod):
    X, Z = prod["X"], prod["Z"]
    for i in range(len(X)):
        assert galois.prod(*X[i]) == Z[i]


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
    assert galois.are_coprime(3, 4, 5) == True
    assert galois.are_coprime(3, 5, 9, 11) == False
    assert galois.are_coprime(2, 3, 7, 256) == False


def test_pow_exceptions():
    with pytest.raises(TypeError):
        galois.pow(1.0, 2, 3)
    with pytest.raises(TypeError):
        galois.pow(1, 2.0, 3)
    with pytest.raises(TypeError):
        galois.pow(1, 2, 3.0)


def test_pow(power):
    X, E, M, Z = power["X"], power["E"], power["M"], power["Z"]
    for i in range(len(X)):
        assert galois.pow(X[i], E[i], M[i]) == Z[i]


def test_isqrt_exceptions():
    with pytest.raises(TypeError):
        galois.isqrt(3.0)
    with pytest.raises(ValueError):
        galois.isqrt(-3)


def test_isqrt(isqrt):
    X, Z = isqrt["X"], isqrt["Z"]
    for i in range(len(X)):
        assert galois.isqrt(X[i]) == Z[i]
