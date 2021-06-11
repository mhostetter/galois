"""
A pytest module to test the functions in math_.py.
"""
import pytest

import galois


def test_isqrt():
    p = galois.mersenne_primes(2000)[-1]
    sqrt_p = galois.isqrt(p)
    assert isinstance(sqrt_p, int)
    assert sqrt_p**2 <= p and not (sqrt_p + 1)**2 <= p

    p = galois.mersenne_primes(2000)[-1] - 1
    sqrt_p = galois.isqrt(p)
    assert isinstance(sqrt_p, int)
    assert sqrt_p**2 <= p and not (sqrt_p + 1)**2 <= p

    p = galois.mersenne_primes(2000)[-1] + 1
    sqrt_p = galois.isqrt(p)
    assert isinstance(sqrt_p, int)
    assert sqrt_p**2 <= p and not (sqrt_p + 1)**2 <= p


def test_iroot():
    p = galois.mersenne_primes(2000)[-1]
    root = galois.iroot(p, 10)
    assert isinstance(root, int)
    assert root**10 <= p and not (root + 1)**10 <= p

    p = galois.mersenne_primes(2000)[-1] - 1
    root = galois.iroot(p, 10)
    assert isinstance(root, int)
    assert root**10 <= p and not (root + 1)**10 <= p

    p = galois.mersenne_primes(2000)[-1] + 1
    root = galois.iroot(p, 10)
    assert isinstance(root, int)
    assert root**10 <= p and not (root + 1)**10 <= p


def test_ilog():
    p = galois.mersenne_primes(2000)[-1]
    exponent = galois.ilog(p, 17)
    assert isinstance(exponent, int)
    assert 17**exponent <= p and not 17**(exponent + 1) <= p

    p = galois.mersenne_primes(2000)[-1] - 1
    exponent = galois.ilog(p, 17)
    assert isinstance(exponent, int)
    assert 17**exponent <= p and not 17**(exponent + 1) <= p

    p = galois.mersenne_primes(2000)[-1] + 1
    exponent = galois.ilog(p, 17)
    assert isinstance(exponent, int)
    assert 17**exponent <= p and not 17**(exponent + 1) <= p


def test_lcm():
    assert galois.lcm() == 1
    assert galois.lcm(1, 0, 4) == 0
    assert galois.lcm(2, 4, 14) == 28

    prime1, prime2 = galois.mersenne_primes(100)[-2:]
    assert galois.lcm(prime1, prime2) == prime1 * prime2


def test_prod():
    assert galois.prod([2, 4, 14]) == 2*4*14
    assert galois.prod([2, 4, 14], start=2) == 2*2*4*14
