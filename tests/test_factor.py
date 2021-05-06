"""
A pytest module to test the functions relating to integer factorization.
"""
import random

import pytest
import numpy as np

import galois


def test_prime_factorization_small():
    x = 8
    PRIMES = [2,]
    EXPONENTS = [3,]
    p, e = galois.prime_factors(x)
    assert np.array_equal(p, PRIMES)
    assert np.array_equal(e, EXPONENTS)

    x = 10
    PRIMES = [2,5]
    EXPONENTS = [1,1]
    p, e = galois.prime_factors(x)
    assert np.array_equal(p, PRIMES)
    assert np.array_equal(e, EXPONENTS)

    x = 11
    PRIMES = [11,]
    EXPONENTS = [1,]
    p, e = galois.prime_factors(x)
    assert np.array_equal(p, PRIMES)
    assert np.array_equal(e, EXPONENTS)


def test_prime_factorization_large():
    x = 3015341941
    PRIMES = [46021,65521]
    EXPONENTS = [1,1]
    p, e = galois.prime_factors(x)
    assert np.array_equal(p, PRIMES)
    assert np.array_equal(e, EXPONENTS)


def test_prime_factorization_extremely_large():
    prime = 1000000000000000035000061
    p, e = galois.prime_factors(prime)
    assert np.array_equal(p, [prime])
    assert np.array_equal(e, [1])

    p, e = galois.prime_factors(prime - 1)
    p = np.array(p, dtype=object)
    e = np.array(e, dtype=object)
    assert np.multiply.reduce(p**e) == prime - 1


def test_smooth():
    assert galois.is_smooth(2**10, 2)
    assert galois.is_smooth(10, 5)
    assert galois.is_smooth(12, 5)
    assert not galois.is_smooth(14, 5)
    assert galois.is_smooth(60**2, 5)
