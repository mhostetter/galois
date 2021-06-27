"""
A pytest module to test the functions relating to integer factorization.
"""
import random

import pytest
import numpy as np

import galois


def test_factors_small():
    assert galois.factors(8) == ([2], [3])
    assert galois.factors(10) == ([2, 5], [1, 1])
    assert galois.factors(11) == ([11], [1])
    assert galois.factors(24) == ([2, 3], [3, 1])


def test_factors_large():
    assert galois.factors(3015341941) == ([46021, 65521], [1, 1])
    assert galois.factors(12345678) == ([2, 3, 47, 14593], [1, 2, 1, 1])


def test_factors_extremely_large():
    assert galois.factors(1000000000000000035000061) == ([1000000000000000035000061], [1])
    assert galois.factors(1000000000000000035000061 - 1) == ([2, 3, 5, 17, 19, 112850813, 457237177399], [2, 1, 1, 1, 1, 1, 1])


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


def test_divisor_sigma():
    # https://oeis.org/A000005
    sigma_0 = [1,2,2,3,2,4,2,4,3,4,2,6,2,4,4,5,2,6,2,6,4,4,2,8,3,4,4,6,2,8,2,6,4,4,4,9,2,4,4,8,2,8,2,6,6,4,2,10,3,6,4,6,2,8,4,8,4,4,2,12,2,4,6,7,4,8,2,6,4,8,2,12,2,4,6,6,4,8,2,10,5,4,2,12,4,4,4,8,2,12,4,6,4,4,4,12,2,6,6,9,2,8,2,8]
    assert [galois.divisor_sigma(n, k=0) for n in range(1, 105)] == sigma_0

    # https://oeis.org/A000203
    sigma_1 = [1,3,4,7,6,12,8,15,13,18,12,28,14,24,24,31,18,39,20,42,32,36,24,60,31,42,40,56,30,72,32,63,48,54,48,91,38,60,56,90,42,96,44,84,78,72,48,124,57,93,72,98,54,120,72,120,80,90,60,168,62,96,104,127,84,144,68,126,96,144]
    assert [galois.divisor_sigma(n, k=1) for n in range(1, 71)] == sigma_1

    # https://oeis.org/A001157
    sigma_2 = [1,5,10,21,26,50,50,85,91,130,122,210,170,250,260,341,290,455,362,546,500,610,530,850,651,850,820,1050,842,1300,962,1365,1220,1450,1300,1911,1370,1810,1700,2210,1682,2500,1850,2562,2366,2650,2210,3410,2451,3255]
    assert [galois.divisor_sigma(n, k=2) for n in range(1, 51)] == sigma_2

    # https://oeis.org/A001158
    sigma_3 = [1,9,28,73,126,252,344,585,757,1134,1332,2044,2198,3096,3528,4681,4914,6813,6860,9198,9632,11988,12168,16380,15751,19782,20440,25112,24390,31752,29792,37449,37296,44226,43344,55261,50654,61740,61544,73710,68922,86688]
    assert [galois.divisor_sigma(n, k=3) for n in range(1, 43)] == sigma_3


def test_is_prime_power():
    # https://oeis.org/A246655
    prime_powers = np.array([2,3,4,5,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,41,43,47,49,53,59,61,64,67,71,73,79,81,83,89,97,101,103,107,109,113,121,125,127,128,131,137,139,149,151,157,163,167,169,173,179,181,191,193,197,199,211])
    n = np.arange(1, prime_powers[-1] + 1)
    is_prime_power = np.zeros(n.size, dtype=bool)
    is_prime_power[prime_powers - 1] = True  # -1 for 1-indexed
    assert [galois.is_prime_power(ni) for ni in n] == is_prime_power.tolist()


def test_is_perfect_power():
    # https://oeis.org/A001597
    perfect_powers = np.array([1,4,8,9,16,25,27,32,36,49,64,81,100,121,125,128,144,169,196,216,225,243,256,289,324,343,361,400,441,484,512,529,576,625,676,729,784,841,900,961,1000,1024,1089,1156,1225,1296,1331,1369,1444,1521,1600,1681,1728,1764])
    n = np.arange(1, perfect_powers[-1] + 1)
    is_perfect_power = np.zeros(n.size, dtype=bool)
    is_perfect_power[perfect_powers - 1] = True  # -1 for 1-indexed
    assert [galois.is_perfect_power(ni) for ni in n] == is_perfect_power.tolist()


def test_is_square_free():
    # https://oeis.org/A005117
    square_frees = np.array([1,2,3,5,6,7,10,11,13,14,15,17,19,21,22,23,26,29,30,31,33,34,35,37,38,39,41,42,43,46,47,51,53,55,57,58,59,61,62,65,66,67,69,70,71,73,74,77,78,79,82,83,85,86,87,89,91,93,94,95,97,101,102,103,105,106,107,109,110,111,113])
    n = np.arange(1, square_frees[-1] + 1)
    is_square_free = np.zeros(n.size, dtype=bool)
    is_square_free[square_frees - 1] = True  # -1 for 1-indexed
    assert [galois.is_square_free(ni) for ni in n] == is_square_free.tolist()


def test_smooth():
    assert galois.is_smooth(2**10, 2)
    assert galois.is_smooth(10, 5)
    assert galois.is_smooth(12, 5)
    assert not galois.is_smooth(14, 5)
    assert galois.is_smooth(60**2, 5)
