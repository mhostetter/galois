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


def test_perfect_power():
    assert galois.perfect_power(5) is None
    assert galois.perfect_power(6) is None
    assert galois.perfect_power(6*16) is None
    assert galois.perfect_power(16*125) is None

    assert galois.perfect_power(9) == (3, 2)
    assert galois.perfect_power(36) == (6, 2)
    assert galois.perfect_power(125) == (5, 3)
    assert galois.perfect_power(216) == (6, 3)


def test_trial_division():
    n = 2**4 * 17**3 * 113 * 15013
    assert galois.trial_division(n) == ([2, 17, 113, 15013], [4, 3, 1, 1], 1)
    assert galois.trial_division(n, B=500) == ([2, 17, 113], [4, 3, 1], 15013)
    assert galois.trial_division(n, B=100) == ([2, 17], [4, 3], 113*15013)


def test_pollard_p1():
    p = 1458757  # p - 1 factors: [2, 3, 13, 1039], [2, 3, 1, 1]
    q = 1326001  # q - 1 factors: [2, 3, 5, 13, 17], [4, 1, 3, 1, 1]
    assert galois.pollard_p1(p*q, 17) is None
    assert galois.pollard_p1(p*q, 125) == q
    assert galois.pollard_p1(p*q, 81, B2=1100) == p

    p = 1598442007  # p - 1 factors: [2, 3, 7, 38058143], [1, 1, 1, 1]
    q = 1316659213  # q - 1 factors: [2, 3, 11, 83, 4451], [2, 4, 1, 1, 1]
    assert galois.pollard_p1(p*q, 83) is None
    assert galois.pollard_p1(p*q, 5000) == q
    assert galois.pollard_p1(p*q, 83, B2=5000) == q

    p = 1636344139  # p - 1 factors: [2, 3, 11, 13, 1381], [1, 1, 1, 1, 2]
    q = 1476638609  # q - 1 factors: [2, 137, 673649], [4, 1, 1]
    assert galois.pollard_p1(p*q, 150) is None
    assert galois.pollard_p1(p*q, 150, B2=10_000) is None
    # assert galois.pollard_p1(p*q, 150, B2=675_000) == q  # TODO: Why doesn't this work?

    n = 2133861346249  # n factors: [37, 41, 5471, 257107], [1, 1, 1, 1]
    # 37 - 1 factors: [2, 3], [2, 2]
    # 41 - 1 factors: [2, 5], [3, 1]
    # 5471 - 1 factors: [2, 5, 547], [1, 1, 1]
    # 257107 - 1 factors: [2, 3, 73, 587], [1, 1, 1, 1]
    assert galois.pollard_p1(n, 10) == 37


def test_pollard_rho():
    p = 1458757
    q = 1326001
    assert galois.pollard_rho(p*q) == p
    assert galois.pollard_rho(p*q, c=4) == q

    p = 1598442007
    q = 1316659213
    assert galois.pollard_rho(p*q) == p
    assert galois.pollard_rho(p*q, c=3) == q

    p = 1636344139
    q = 1476638609
    assert galois.pollard_rho(p*q) == q
    assert galois.pollard_rho(p*q, c=6) == p


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


def test_is_smooth():
    # https://oeis.org/A000079
    smooths = np.array([1,2,4,8,16,32,64,128,256,512,1024])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 2) for ni in n] == is_smooth.tolist()

    # https://oeis.org/A003586
    smooths = np.array([1,2,3,4,6,8,9,12,16,18,24,27,32,36,48,54,64,72,81,96,108,128,144,162,192,216,243,256,288,324,384,432,486,512,576,648,729,768,864,972,1024])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 3) for ni in n] == is_smooth.tolist()

    # https://oeis.org/A051037
    smooths = np.array([1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,45,48,50,54,60,64,72,75,80,81,90,96,100,108,120,125,128,135,144,150,160,162,180,192,200,216,225,240,243,250,256,270,288,300,320,324,360,375,384,400,405])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 5) for ni in n] == is_smooth.tolist()

    # https://oeis.org/A002473
    smooths = np.array([1,2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,25,27,28,30,32,35,36,40,42,45,48,49,50,54,56,60,63,64,70,72,75,80,81,84,90,96,98,100,105,108,112,120,125,126,128,135,140,144,147,150,160,162,168,175,180,189,192])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 7) for ni in n] == is_smooth.tolist()

    # https://oeis.org/A018336
    smooths = np.array([1,2,3,5,6,7,10,14,15,21,30,35,42,70,105,210])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 7) and galois.is_square_free(ni) for ni in n] == is_smooth.tolist()

    # https://oeis.org/A051038
    smooths = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,18,20,21,22,24,25,27,28,30,32,33,35,36,40,42,44,45,48,49,50,54,55,56,60,63,64,66,70,72,75,77,80,81,84,88,90,96,98,99,100,105,108,110,112,120,121,125,126,128,132,135,140])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 11) for ni in n] == is_smooth.tolist()

    # https://oeis.org/A051038
    smooths = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,18,20,21,22,24,25,27,28,30,32,33,35,36,40,42,44,45,48,49,50,54,55,56,60,63,64,66,70,72,75,77,80,81,84,88,90,96,98,99,100,105,108,110,112,120,121,125,126,128,132,135,140])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 11) for ni in n] == is_smooth.tolist()

    # https://oeis.org/A087005
    smooths = np.array([1,2,3,5,6,7,10,11,14,15,21,22,30,33,35,42,55,66,70,77,105,110,154,165,210,231,330,385,462,770,1155,2310])
    n = np.arange(1, smooths[-1] + 1)
    is_smooth = np.zeros(n.size, dtype=bool)
    is_smooth[smooths - 1] = True  # -1 for 1-indexed
    assert [galois.is_smooth(ni, 11) and galois.is_square_free(ni) for ni in n] == is_smooth.tolist()


def test_is_powersmooth():
    assert galois.is_powersmooth(2**4 * 3**2 * 5, 5) == False
    assert galois.is_powersmooth(2**4 * 3**2 * 5, 9) == False
    assert galois.is_powersmooth(2**4 * 3**2 * 5, 16) == True

    assert galois.is_powersmooth(2**4 * 3**2 * 5*3, 5) == False
    assert galois.is_powersmooth(2**4 * 3**2 * 5*3, 25) == False
    assert galois.is_powersmooth(2**4 * 3**2 * 5*3, 125) == True

    assert galois.is_powersmooth(13 * 23, 4) == False

    for n in range(2, 1_000):
        p, e = galois.factors(n)
        assert all([pi**ei <= 50 for pi, ei in zip(p, e)]) == galois.is_powersmooth(n, 50)
