"""
A pytest module to test the various algorithms in the galois package.
"""
import math
import random

import numpy as np
import pytest

import galois


def test_factors():
    factors = galois.factors(120)
    true_factors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
    assert np.array_equal(factors, true_factors)


def test_euclidean_algorithm():
    a = random.randint(0, 1_000_000)
    b = random.randint(0, 1_000_000)
    gcd = galois.euclidean_algorithm(a, b)
    assert gcd == math.gcd(a, b)


def test_extended_euclidean_algorithm():
    a = random.randint(0, 1_000_000)
    b = random.randint(0, 1_000_000)
    x, y, gcd = galois.extended_euclidean_algorithm(a, b)
    assert gcd == math.gcd(a, b)
    assert a*x + b*y == gcd


def test_chinese_remainder_theorem():
    a = [0, 3, 4]
    m = [3, 4, 5]
    x = galois.chinese_remainder_theorem(a, m)
    assert x == 39
    for i in range(len(a)):
        assert x % m[i] == a[i]


def test_euler_totient():
    # https://oeis.org/A000010
    N = list(range(1,70))
    PHI = [1,1,2,2,4,2,6,4,6,4,10,4,12,6,8,8,16,6,18,8,12,10,22,8,20,12,18,12,28,8,30,16,20,16,24,12,36,18,24,16,40,12,42,20,24,22,46,16,42,20,32,24,52,18,40,24,36,28,58,16,60,30,36,32,48,20,66,32,44]
    for n, phi in zip(N, PHI):
        assert galois.euler_totient(n) == phi


def test_carmichael():
    # https://oeis.org/A002322
    N = list(range(1,82))
    LAMBDA = [1,1,2,2,4,2,6,2,6,4,10,2,12,6,4,4,16,6,18,4,6,10,22,2,20,12,18,6,28,4,30,8,10,16,12,6,36,18,12,4,40,6,42,10,12,22,46,4,42,20,16,12,52,18,20,6,18,28,58,4,60,30,6,16,12,10,66,16,22,12,70,6,72,36,20,18,30,12,78,4,54]
    for n, lambda_ in zip(N, LAMBDA):
        assert galois.carmichael(n) == lambda_


def test_modular_exp():
    galois.modular_exp(2, 0, 71) == 1
    galois.modular_exp(1, 4, 71) == 1
    galois.modular_exp(2, 4, 71) == 16
    galois.modular_exp(5, 3, 71) == 4

    # https://users.cs.jmu.edu/abzugcx/Public/Discrete-Structures-II/Modular-Exponentiation.pdf
    galois.modular_exp(23, 20, 29) == 24
    galois.modular_exp(23, 391, 55) == 12
    galois.modular_exp(31, 397, 29) == 26


def test_modular_exp_large_integers():
    base = 123456789123456789
    exponent = 1234
    modulus = 98765431
    assert galois.modular_exp(base, exponent, modulus) == base**exponent % modulus


def test_primitive_root():
    # https://oeis.org/A001918
    primes = galois._prime.PRIMES
    ns = list(range(1,105))
    roots = [1,2,2,3,2,2,3,2,5,2,3,2,6,3,5,2,2,2,2,7,5,3,2,3,5,2,5,2,6,3,3,2,3,2,2,6,5,2,5,2,2,2,19,5,2,3,2,3,2,6,3,7,7,6,3,5,2,6,5,3,3,2,5,17,10,2,3,10,2,2,3,7,6,2,2,5,2,5,3,21,2,2,7,5,15,2,3,13,2,3,2,13,3,2,7,5,2,3,2,2,2,2,2,3]
    for n, root in zip(ns, roots):
        assert galois.primitive_root(primes[n-1]) == root


# def test_sum_primitive_roots():
#     # https://oeis.org/A088144
#     primes = galois._prime.PRIMES
#     ns = list(range(1,46))
#     sums = [1,2,5,8,23,26,68,57,139,174,123,222,328,257,612,636,886,488,669,1064,876,1105,1744,1780,1552,2020,1853,2890,1962,2712,2413,3536,4384,3335,5364,3322,3768,4564,7683,7266,8235,4344,8021,6176,8274]
#     for n, sum in zip(ns, sums):
#         roots = galois.primitive_roots(primes[n-1])
#         assert np.sum(np.array(roots)) == sum


# def test_primitive_roots():
#     # https://en.wikipedia.org/wiki/Primitive_root_modulo_n
#     assert galois.primitive_roots(1) == [0]
#     assert galois.primitive_roots(2) == [1]
#     assert galois.primitive_roots(3) == [2]
#     assert galois.primitive_roots(4) == [3]
#     assert galois.primitive_roots(5) == [2,3]
#     assert galois.primitive_roots(6) == [5]
#     assert galois.primitive_roots(7) == [3,5]
#     assert galois.primitive_roots(8) == []
#     assert galois.primitive_roots(9) == [2,5]
#     assert galois.primitive_roots(10) == [3,7]
#     assert galois.primitive_roots(11) == [2,6,7,8]
#     assert galois.primitive_roots(12) == []
#     assert galois.primitive_roots(13) == [2,6,7,11]
#     assert galois.primitive_roots(14) == [3,5]
#     assert galois.primitive_roots(15) == []

#     assert galois.primitive_roots(29) == [2,3,8,10,11,14,15,18,19,21,26,27]
#     assert galois.primitive_roots(31) == [3,11,12,13,17,21,22,24]
#     assert galois.primitive_roots(61) == [2,6,7,10,17,18,26,30,31,35,43,44,51,54,55,59]
