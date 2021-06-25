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


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 31, 13*7, 120, 120*7])
def test_divisors(n):
    assert galois.divisors(n) == [d for d in range(1, n + 1) if n % d == 0]
    assert galois.divisors(-n) == [d for d in range(1, abs(n) + 1) if n % d == 0]


def test_divisors_random():
    for _ in range(10):
        n = random.randint(2, 10_000)
        assert galois.divisors(n) == [d for d in range(1, n + 1) if n % d == 0]
        assert galois.divisors(-n) == [d for d in range(1, abs(n) + 1) if n % d == 0]


def test_divisors_number():
    # https://oeis.org/A000005
    d_n = [1,2,2,3,2,4,2,4,3,4,2,6,2,4,4,5,2,6,2,6,4,4,2,8,3,4,4,6,2,8,2,6,4,4,4,9,2,4,4,8,2,8,2,6,6,4,2,10,3,6,4,6,2,8,4,8,4,4,2,12,2,4,6,7,4,8,2,6,4,8,2,12,2,4,6,6,4,8,2,10,5,4,2,12,4,4,4,8,2,12,4,6,4,4,4,12,2,6,6,9,2,8,2,8]
    assert [len(galois.divisors(n)) for n in range(1, 105)] == d_n


def test_smooth():
    assert galois.is_smooth(2**10, 2)
    assert galois.is_smooth(10, 5)
    assert galois.is_smooth(12, 5)
    assert not galois.is_smooth(14, 5)
    assert galois.is_smooth(60**2, 5)
