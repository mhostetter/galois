"""
A pytest module to test the algorithms in galois/algorithm.py.
"""
import pytest
import numpy as np

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


def test_euler_totient():
    # https://oeis.org/A000010
    N = list(range(1,70))
    PHI = [1,1,2,2,4,2,6,4,6,4,10,4,12,6,8,8,16,6,18,8,12,10,22,8,20,12,18,12,28,8,30,16,20,16,24,12,36,18,24,16,40,12,42,20,24,22,46,16,42,20,32,24,52,18,40,24,36,28,58,16,60,30,36,32,48,20,66,32,44]
    for n, phi in zip(N, PHI):
        assert galois.euler_totient(n) == phi


def test_carmichael():
    # https://oeis.org/A002322/list
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


def test_primitive_roots():
    # https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    assert galois.primitive_roots(1) == [0]
    assert galois.primitive_roots(2) == [1]
    assert galois.primitive_roots(3) == [2]
    assert galois.primitive_roots(4) == [3]
    assert galois.primitive_roots(5) == [2,3]
    assert galois.primitive_roots(6) == [5]
    assert galois.primitive_roots(7) == [3,5]
    assert galois.primitive_roots(8) == []
    assert galois.primitive_roots(9) == [2,5]
    assert galois.primitive_roots(10) == [3,7]
    assert galois.primitive_roots(11) == [2,6,7,8]
    assert galois.primitive_roots(12) == []
    assert galois.primitive_roots(13) == [2,6,7,11]
    assert galois.primitive_roots(14) == [3,5]
    assert galois.primitive_roots(15) == []

    assert galois.primitive_roots(29) == [2,3,8,10,11,14,15,18,19,21,26,27]
    assert galois.primitive_roots(31) == [3,11,12,13,17,21,22,24]
    assert galois.primitive_roots(61) == [2,6,7,10,17,18,26,30,31,35,43,44,51,54,55,59]
