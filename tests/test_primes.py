"""
A pytest module to test the functions relating to primes.
"""
import random

import numpy as np
import pytest

import galois


def test_prev_next_prime():
    assert galois.prev_prime(8) == 7
    assert galois.next_prime(8) == 11

    assert galois.prev_prime(11) == 11
    assert galois.next_prime(11) == 13


def test_prime_factorization_small():
    x = 8
    P = [2,]
    K = [3,]
    p, k = galois.prime_factors(x)
    assert np.all(p == P)
    assert np.all(k == K)

    x = 10
    P = [2,5]
    K = [1,1]
    p, k = galois.prime_factors(x)
    assert np.all(p == P)
    assert np.all(k == K)

    x = 11
    P = [11,]
    K = [1,]
    p, k = galois.prime_factors(x)
    assert np.all(p == P)
    assert np.all(k == K)


def test_prime_factorization_large():
    x = 3015341941
    P = [46021,65521]
    K = [1,1]
    p, k = galois.prime_factors(x)
    assert np.all(p == P)
    assert np.all(k == K)


def test_prime_factorization_extremely_large():
    prime = 1000000000000000035000061
    p, k = galois.prime_factors(prime)
    assert np.all(p == [prime])
    assert np.all(k == [1])

    p, k = galois.prime_factors(prime - 1)
    assert np.multiply.reduce(p**k) == prime - 1


def test_fermat_primality_test_on_primes():
    primes = random.choices(galois._prime.PRIMES, k=10)
    for prime in primes:
        # Fermat's primality test should never call a prime a composite
        assert galois.fermat_primality_test(prime) == True


def test_fermat_primality_test_on_pseudoprimes():
    # https://oeis.org/A001262
    pseudoprimes = [2047,3277,4033,4681,8321,15841,29341,42799,49141,52633,65281,74665,80581,85489,88357,90751,104653,130561,196093,220729,233017,252601,253241,256999,271951,280601,314821,357761,390937,458989,476971,486737]
    for pseudoprime in pseudoprimes:
        # Fermat's primality test is fooled by strong pseudoprimes
        assert galois.fermat_primality_test(pseudoprime) == True


def test_miller_rabin_primality_test_on_primes():
    primes = random.choices(galois._prime.PRIMES, k=10)
    for prime in primes:
        # Miller-Rabin's primality test should never call a prime a composite
        assert galois.fermat_primality_test(prime) == True


def test_miller_rabin_primality_test_on_pseudoprimes():
    # https://oeis.org/A001262
    pseudoprimes = [2047,3277,4033,4681,8321,15841,29341,42799,49141,52633,65281,74665,80581,85489,88357,90751,104653,130561,196093,220729,233017,252601,253241,256999,271951,280601,314821,357761,390937,458989,476971,486737]
    for pseudoprime in pseudoprimes:
        # With sufficient rounds, the Miller-Rabin primality test will detect the pseudoprimes as composite
        assert galois.miller_rabin_primality_test(pseudoprime, rounds=7) == False
