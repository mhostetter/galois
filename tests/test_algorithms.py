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


def test_is_cyclic():
    assert galois.is_cyclic(2) == True
    assert galois.is_cyclic(4) == True
    assert galois.is_cyclic(2 * 3**3) == True
    assert galois.is_cyclic(5**7) == True
    assert galois.is_cyclic(2**3) == False
    assert galois.is_cyclic(2*5*7*9) == False


def test_primitive_root_non_integer():
    with pytest.raises(TypeError):
        galois.primitive_root(13.0)


def test_primitive_root_non_positive_integer():
    with pytest.raises(ValueError):
        galois.primitive_root(0)
    with pytest.raises(ValueError):
        galois.primitive_root(-2)


def test_primitive_root_invalid_start():
    with pytest.raises(ValueError):
        galois.primitive_root(7, start=0)
    with pytest.raises(ValueError):
        galois.primitive_root(7, start=7)


def test_primitive_root_invalid_stop():
    with pytest.raises(ValueError):
        galois.primitive_root(7, stop=1)
    with pytest.raises(ValueError):
        galois.primitive_root(7, start=6, stop=6)


def test_smallest_primitive_root():
    # https://oeis.org/A046145
    ns = range(1, 101)
    roots = [0,1,2,3,2,5,3,None,2,3,2,None,2,3,None,None,3,5,2,None,None,7,5,None,2,7,2,None,2,None,3,None,None,3,None,None,2,3,None,None,6,None,3,None,None,5,5,None,3,3,None,None,2,5,None,None,None,3,2,None,2,3,None,None,None,None,2,None,None,None,7,None,5,5,None,None,None,None,3,None,2,7,2,None,None,3,None,None,3,None,None,None,None,5,None,None,5,3,None,None]
    for n, root in zip(ns, roots):
        assert galois.primitive_root(n) == root


def test_largest_primitive_root():
    # https://oeis.org/A046146
    ns = range(1, 101)
    roots = [0,1,2,3,3,5,5,None,5,7,8,None,11,5,None,None,14,11,15,None,None,19,21,None,23,19,23,None,27,None,24,None,None,31,None,None,35,33,None,None,35,None,34,None,None,43,45,None,47,47,None,None,51,47,None,None,None,55,56,None,59,55,None,None,None,None,63,None,None,None,69,None,68,69,None,None,None,None,77,None,77,75,80,None,None]
    for n, root in zip(ns, roots):
        assert galois.primitive_root(n, largest=True) == root


def test_smallest_primitive_root_of_primes():
    # https://oeis.org/A001918
    primes = galois._prime.PRIMES
    ns = list(range(1,105))
    roots = [1,2,2,3,2,2,3,2,5,2,3,2,6,3,5,2,2,2,2,7,5,3,2,3,5,2,5,2,6,3,3,2,3,2,2,6,5,2,5,2,2,2,19,5,2,3,2,3,2,6,3,7,7,6,3,5,2,6,5,3,3,2,5,17,10,2,3,10,2,2,3,7,6,2,2,5,2,5,3,21,2,2,7,5,15,2,3,13,2,3,2,13,3,2,7,5,2,3,2,2,2,2,2,3]
    for n, root in zip(ns, roots):
        assert galois.primitive_root(primes[n-1]) == root


def test_largest_primitive_root_of_primes():
    # https://oeis.org/A071894
    primes = galois._prime.PRIMES
    ns = list(range(1,60))
    roots = [1,2,3,5,8,11,14,15,21,27,24,35,35,34,45,51,56,59,63,69,68,77,80,86,92,99,101,104,103,110,118,128,134,135,147,146,152,159,165,171,176,179,189,188,195,197,207,214,224,223,230,237,234,248,254,261,267,269,272]
    for n, root in zip(ns, roots):
        assert galois.primitive_root(primes[n-1], largest=True) == root


def test_primitive_roots():
    # https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    assert list(galois.primitive_roots(1)) == [0]
    assert list(galois.primitive_roots(2)) == [1]
    assert list(galois.primitive_roots(3)) == [2]
    assert list(galois.primitive_roots(4)) == [3]
    assert list(galois.primitive_roots(5)) == [2,3]
    assert list(galois.primitive_roots(6)) == [5]
    assert list(galois.primitive_roots(7)) == [3,5]
    assert list(galois.primitive_roots(8)) == []
    assert list(galois.primitive_roots(9)) == [2,5]
    assert list(galois.primitive_roots(10)) == [3,7]
    assert list(galois.primitive_roots(11)) == [2,6,7,8]
    assert list(galois.primitive_roots(12)) == []
    assert list(galois.primitive_roots(13)) == [2,6,7,11]
    assert list(galois.primitive_roots(14)) == [3,5]
    assert list(galois.primitive_roots(15)) == []
    assert list(galois.primitive_roots(16)) == []
    assert list(galois.primitive_roots(17)) == [3,5,6,7,10,11,12,14]
    assert list(galois.primitive_roots(18)) == [5,11]
    assert list(galois.primitive_roots(19)) == [2,3,10,13,14,15]
    assert list(galois.primitive_roots(20)) == []
    assert list(galois.primitive_roots(21)) == []
    assert list(galois.primitive_roots(22)) == [7,13,17,19]
    assert list(galois.primitive_roots(23)) == [5,7,10,11,14,15,17,19,20,21]
    assert list(galois.primitive_roots(24)) == []
    assert list(galois.primitive_roots(25)) == [2,3,8,12,13,17,22,23]
    assert list(galois.primitive_roots(26)) == [7,11,15,19]
    assert list(galois.primitive_roots(27)) == [2,5,11,14,20,23]
    assert list(galois.primitive_roots(28)) == []
    assert list(galois.primitive_roots(29)) == [2,3,8,10,11,14,15,18,19,21,26,27]
    assert list(galois.primitive_roots(30)) == []


def test_number_of_primitive_roots():
    # https://oeis.org/A046144
    ns = list(range(1,92))
    num_roots = [1,1,1,1,2,1,2,0,2,2,4,0,4,2,0,0,8,2,6,0,0,4,10,0,8,4,6,0,12,0,8,0,0,8,0,0,12,6,0,0,16,0,12,0,0,10,22,0,12,8,0,0,24,6,0,0,0,12,28,0,16,8,0,0,0,0,20,0,0,0,24,0,24,12,0,0,0,0,24,0,18,16,40,0,0,12,0,0,40,0,0]
    for n, num in zip(ns, num_roots):
        assert len(list(galois.primitive_roots(n))) == num


def test_number_of_primitive_roots_of_primes():
    # https://oeis.org/A008330
    primes = galois._prime.PRIMES
    ns = list(range(1,71))
    num_roots = [1,1,2,2,4,4,8,6,10,12,8,12,16,12,22,24,28,16,20,24,24,24,40,40,32,40,32,52,36,48,36,48,64,44,72,40,48,54,82,84,88,48,72,64,84,60,48,72,112,72,112,96,64,100,128,130,132,72,88,96,92,144,96,120,96,156,80,96,172,112]
    for n, num in zip(ns, num_roots):
        assert len(list(galois.primitive_roots(primes[n-1]))) == num


def test_sum_primitive_roots_of_primes():
    # https://oeis.org/A088144
    primes = galois._prime.PRIMES
    ns = list(range(1,46))
    sums = [1,2,5,8,23,26,68,57,139,174,123,222,328,257,612,636,886,488,669,1064,876,1105,1744,1780,1552,2020,1853,2890,1962,2712,2413,3536,4384,3335,5364,3322,3768,4564,7683,7266,8235,4344,8021,6176,8274]
    for n, s in zip(ns, sums):
        assert sum(list(galois.primitive_roots(primes[n-1]))) == s


@pytest.mark.parametrize("n", [2, 4, 7**2, 2*257])
def test_primitive_roots_are_generators(n):
    n = int(n)
    congruences = [a for a in range(1, n) if math.gcd(n, a) == 1]
    phi = galois.euler_totient(n)
    assert len(congruences) == phi

    roots = list(galois.primitive_roots(n))
    for root in roots:
        elements = [pow(root, i, n) for i in range(1, n)]
        assert set(congruences) == set(elements)

    assert len(roots) == galois.euler_totient(phi)
