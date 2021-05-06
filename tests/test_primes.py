"""
A pytest module to test the functions relating to primes.
"""
import random

import pytest
import numpy as np

import galois


def test_primes():
    assert galois.primes(19) == [2, 3, 5, 7, 11, 13, 17, 19]
    assert galois.primes(20) == [2, 3, 5, 7, 11, 13, 17, 19]

    with pytest.raises(TypeError):
        galois.primes(20.0)
    with pytest.raises(ValueError):
        galois.primes(1)


def test_kth_prime():
    assert galois.kth_prime(1) == 2
    assert galois.kth_prime(2) == 3
    assert galois.kth_prime(100) == 541
    assert galois.kth_prime(1000) == 7919

    with pytest.raises(TypeError):
        galois.kth_prime(20.0)
    with pytest.raises(ValueError):
        galois.kth_prime(0)
    with pytest.raises(ValueError):
        galois.kth_prime(galois.prime.MAX_K + 1)


def test_prev_prime():
    assert galois.prev_prime(8) == 7
    assert galois.prev_prime(11) == 11

    with pytest.raises(TypeError):
        galois.prev_prime(20.0)
    with pytest.raises(ValueError):
        galois.prev_prime(1)
    with pytest.raises(ValueError):
        galois.prev_prime(galois.prime.MAX_PRIME + 1)


def test_next_prime():
    assert galois.next_prime(8) == 11
    assert galois.next_prime(11) == 13

    with pytest.raises(TypeError):
        galois.next_prime(20.0)
    with pytest.raises(ValueError):
        galois.next_prime(galois.prime.MAX_PRIME)


def test_mersenne_exponents():
    # https://oeis.org/A000043
    exponents = [2,3,5,7,13,17,19,31,61,89,107,127]  # Up to 128 bits
    assert galois.mersenne_exponents(128) == exponents


def test_mersenne_primes():
    # https://oeis.org/A000668
    primes = [3,7,31,127,8191,131071,524287,2147483647,2305843009213693951,618970019642690137449562111,162259276829213363391578010288127,170141183460469231731687303715884105727]  # Up to 128 bits
    assert galois.mersenne_primes(128) == primes


def test_fermat_primality_test_on_primes():
    primes = random.choices(galois.prime.PRIMES, k=10)
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
    primes = random.choices(galois.prime.PRIMES, k=10)
    for prime in primes:
        # Miller-Rabin's primality test should never call a prime a composite
        assert galois.fermat_primality_test(prime) == True


def test_miller_rabin_primality_test_on_pseudoprimes():
    # https://oeis.org/A001262
    pseudoprimes = [2047,3277,4033,4681,8321,15841,29341,42799,49141,52633,65281,74665,80581,85489,88357,90751,104653,130561,196093,220729,233017,252601,253241,256999,271951,280601,314821,357761,390937,458989,476971,486737]
    for pseudoprime in pseudoprimes:
        # With sufficient rounds, the Miller-Rabin primality test will detect the pseudoprimes as composite
        assert galois.miller_rabin_primality_test(pseudoprime, rounds=7) == False
