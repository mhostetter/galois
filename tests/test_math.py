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

    sqrt_p = galois.isqrt(p - 1)
    assert isinstance(sqrt_p, int)
    assert sqrt_p**2 <= p and not (sqrt_p + 1)**2 <= p

    sqrt_p = galois.isqrt(p + 1)
    assert isinstance(sqrt_p, int)
    assert sqrt_p**2 <= p and not (sqrt_p + 1)**2 <= p


def test_lcm():
    assert galois.lcm() == 1
    assert galois.lcm(1, 0, 4) == 0
    assert galois.lcm(2, 4, 14) == 28

    prime1, prime2 = galois.mersenne_primes(100)[-2:]
    assert galois.lcm(prime1, prime2) == prime1 * prime2
