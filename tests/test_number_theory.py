"""
A pytest module to test number theoretic functions.
"""
import pytest
import numpy as np

import galois


###############################################################################
# Divisibility
###############################################################################

def test_gcd_exceptions():
    with pytest.raises(TypeError):
        galois.gcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.gcd(10, 12.0)


def test_gcd(egcd):
    X, Y, D = egcd["X"], egcd["Y"], egcd["D"]
    for i in range(len(X)):
        assert galois.gcd(X[i], Y[i]) == D[i]


def test_egcd_exceptions():
    with pytest.raises(TypeError):
        galois.egcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.egcd(10, 12.0)


def test_egcd(egcd):
    X, Y, D, S, T = egcd["X"], egcd["Y"], egcd["D"], egcd["S"], egcd["T"]
    for i in range(len(X)):
        assert galois.egcd(X[i], Y[i]) == (D[i], S[i], T[i])


def test_lcm_exceptions():
    with pytest.raises(TypeError):
        galois.lcm(1.0, 2, 3)
    with pytest.raises(TypeError):
        galois.lcm(1, 2.0, 3)
    with pytest.raises(TypeError):
        galois.lcm(1, 2, 3.0)
    with pytest.raises(ValueError):
        galois.lcm()


def test_lcm(lcm):
    X, Z = lcm["X"], lcm["Z"]
    for i in range(len(X)):
        assert galois.lcm(*X[i]) == Z[i]


def test_prod_exceptions():
    with pytest.raises(TypeError):
        galois.prod(1, 2.0, 3)
    with pytest.raises(TypeError):
        galois.prod(1, 2, 3.0)
    with pytest.raises(ValueError):
        galois.prod()


def test_prod(prod):
    X, Z = prod["X"], prod["Z"]
    for i in range(len(X)):
        assert galois.prod(*X[i]) == Z[i]


def test_euler_phi_exceptions():
    with pytest.raises(TypeError):
        galois.euler_phi(20.0)
    with pytest.raises(ValueError):
        galois.euler_phi(-1)


def test_euler_phi():
    # https://oeis.org/A000010
    N = list(range(1,70))
    PHI = [1,1,2,2,4,2,6,4,6,4,10,4,12,6,8,8,16,6,18,8,12,10,22,8,20,12,18,12,28,8,30,16,20,16,24,12,36,18,24,16,40,12,42,20,24,22,46,16,42,20,32,24,52,18,40,24,36,28,58,16,60,30,36,32,48,20,66,32,44]
    for n, phi in zip(N, PHI):
        assert galois.euler_phi(n) == phi


def test_totatives_exceptions():
    with pytest.raises(TypeError):
        galois.totatives(20.0)
    with pytest.raises(ValueError):
        galois.totatives(-1)


def test_totatives():
    # https://oeis.org/A000010
    N = list(range(1,70))
    PHI = [1,1,2,2,4,2,6,4,6,4,10,4,12,6,8,8,16,6,18,8,12,10,22,8,20,12,18,12,28,8,30,16,20,16,24,12,36,18,24,16,40,12,42,20,24,22,46,16,42,20,32,24,52,18,40,24,36,28,58,16,60,30,36,32,48,20,66,32,44]
    for n, phi in zip(N, PHI):
        assert len(galois.totatives(n)) == phi


def test_are_coprime_exceptions():
    with pytest.raises(TypeError):
        galois.are_coprime(3.0, 4, 5)
    with pytest.raises(TypeError):
        galois.are_coprime(3, 4.0, 5)
    with pytest.raises(TypeError):
        galois.are_coprime(3, 4, 5.0)
    with pytest.raises(ValueError):
        galois.are_coprime()


def test_are_coprime():
    assert galois.are_coprime(3, 4, 5) == True
    assert galois.are_coprime(3, 5, 9, 11) == False
    assert galois.are_coprime(2, 3, 7, 256) == False


###############################################################################
# Congruences
###############################################################################

def test_pow_exceptions():
    with pytest.raises(TypeError):
        galois.pow(1.0, 2, 3)
    with pytest.raises(TypeError):
        galois.pow(1, 2.0, 3)
    with pytest.raises(TypeError):
        galois.pow(1, 2, 3.0)


def test_pow():
    """
    Sage:
        lut = []
        for _ in range(20):
            base = randint(0, 1_000_000)
            exponent = randint(0, 1_000_000)
            modulus = randint(0, 1_000_000)
            result = pow(base, exponent, modulus)
            lut.append((base, exponent, modulus, result))
        print(lut)
    """
    LUT = [(323434, 327726, 742438, 162670), (841205, 251699, 237735, 93650), (271693, 58834, 905805, 872344), (834391, 494506, 577332, 217513), (424309, 133637, 715490, 51069), (213728, 778789, 346929, 71384), (134584, 118281, 161814, 3850), (387825, 382708, 743837, 72495), (617165, 759526, 997656, 10033), (372370, 792424, 309363, 216235), (98499, 489843, 965820, 24279), (899638, 732171, 888788, 233020), (939745, 576108, 981125, 383250), (623885, 605949, 128282, 105517), (447582, 848997, 724778, 200148), (360331, 355674, 472506, 469429), (85386, 299738, 228229, 40864), (95024, 360241, 488753, 11851), (471190, 788675, 314450, 291400), (49670, 61752, 755248, 57552)]
    for item in LUT:
        base, exponent, modulus = item[0:3]
        result = item[-1]
        assert galois.pow(base, exponent, modulus) == result


def test_crt_exceptions():
    with pytest.raises(TypeError):
        galois.crt(np.array([0, 3, 4]), [3, 4, 5])
    with pytest.raises(TypeError):
        galois.crt([0, 3, 4], np.array([3, 4, 5]))
    with pytest.raises(TypeError):
        galois.crt([0, 3.0, 4], [3, 4, 5])
    with pytest.raises(TypeError):
        galois.crt([0, 3, 4], [3, 4.0, 5])
    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 5, 7])
    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 6])


def test_crt():
    """
    Sage:
        lut = []
        for _ in range(20):
            N = randint(2, 6)
            a = [randint(0, 1_000) for _ in range(N)]
            m = []
            while len(m) < N:
                mi = next_prime(randint(0, 1_000))
                if mi not in m:
                    m.append(mi)
            x = crt(a, m)
            lut.append((a, m, x))
        print(lut)
    """
    LUT = [([975, 426, 300, 372, 596, 856], [457, 331, 521, 701, 71, 907], 1139408681764819), ([85, 653, 323, 655], [331, 479, 601, 191], 10711106463), ([589, 538, 501], [347, 541, 947], 155375738), ([788, 821, 414], [673, 331, 149], 20497003), ([269, 703, 436, 641, 616], [929, 293, 541, 467, 853], 39214084831996), ([270, 190], [173, 277], 15148), ([518, 809, 857, 118], [349, 821, 937, 157], 38154123633), ([711, 735, 1000, 426, 522], [149, 281, 293, 97, 37], 43994914384), ([104, 722, 168, 478], [193, 977, 211, 607], 23841886088), ([64, 160, 626, 702, 883], [877, 907, 251, 307, 839], 6612150797141), ([428, 570, 418, 346, 436], [467, 541, 373, 907, 179], 14825927170624), ([904, 14, 690, 585], [577, 907, 223, 967], 1713097171), ([238, 213, 368, 909, 455, 995], [137, 359, 947, 463, 937, 113], 1425637682359276), ([624, 95, 467, 472, 447, 849], [439, 79, 757, 269, 449, 293], 358511203165372), ([692, 245, 191, 101, 992, 267], [197, 367, 419, 139, 233, 593], 528226613934229), ([767, 794, 410], [331, 727, 359], 35029835), ([938, 992], [547, 17], 1485), ([337, 286, 308, 602, 386, 855], [67, 241, 167, 113, 211, 659], 36856037086592), ([681, 418], [997, 739], 630785), ([897, 343, 555, 245], [701, 89, 827, 379], 10082200693)]
    for item in LUT:
        a, m, x = item
        assert galois.crt(a, m) == x


def test_carmichael_lambda_exceptions():
    with pytest.raises(TypeError):
        galois.carmichael_lambda(20.0)
    with pytest.raises(ValueError):
        galois.carmichael_lambda(-1)


def test_carmichael_lambda():
    # https://oeis.org/A002322
    N = list(range(1,82))
    LAMBDA = [1,1,2,2,4,2,6,2,6,4,10,2,12,6,4,4,16,6,18,4,6,10,22,2,20,12,18,6,28,4,30,8,10,16,12,6,36,18,12,4,40,6,42,10,12,22,46,4,42,20,16,12,52,18,20,6,18,28,58,4,60,30,6,16,12,10,66,16,22,12,70,6,72,36,20,18,30,12,78,4,54]
    for n, lambda_ in zip(N, LAMBDA):
        assert galois.carmichael_lambda(n) == lambda_


def test_legendre_symbol_exceptions():
    with pytest.raises(TypeError):
        galois.legendre_symbol(3.0, 7)
    with pytest.raises(TypeError):
        galois.legendre_symbol(3, 7.0)
    with pytest.raises(ValueError):
        galois.legendre_symbol(3, 4)
    with pytest.raises(ValueError):
        galois.legendre_symbol(3, 2)


def test_legendre_symbol():
    # https://oeis.org/A102283
    assert [galois.legendre_symbol(n, 3) for n in range(0, 105)] == [0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1]
    # https://oeis.org/A080891
    assert [galois.legendre_symbol(n, 5) for n in range(0, 101)] == [0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0]
    # https://oeis.org/A175629
    assert [galois.legendre_symbol(n, 7) for n in range(0, 87)] == [0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1]
    # https://oeis.org/A011582
    assert [galois.legendre_symbol(n, 11) for n in range(0, 81)] == [0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1]
    # https://oeis.org/A011583
    assert [galois.legendre_symbol(n, 13) for n in range(0, 81)] == [0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1]
    # ...
    # https://oeis.org/A165573
    assert [galois.legendre_symbol(n, 257) for n in range(0, 82)] == [0,1,1,-1,1,-1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1]
    # https://oeis.org/A165574
    assert [galois.legendre_symbol(n, 263) for n in range(0, 84)] == [0,1,1,1,1,-1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,-1,1]


def test_jacobi_symbol_exceptions():
    with pytest.raises(TypeError):
        galois.jacobi_symbol(3.0, 7)
    with pytest.raises(TypeError):
        galois.jacobi_symbol(3, 7.0)
    with pytest.raises(ValueError):
        galois.jacobi_symbol(3, 4)
    with pytest.raises(ValueError):
        galois.jacobi_symbol(3, 1)


def test_jacobi_symbol():
    # https://oeis.org/A102283
    assert [galois.jacobi_symbol(n, 3) for n in range(0, 105)] == [0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1]
    # https://oeis.org/A080891
    assert [galois.jacobi_symbol(n, 5) for n in range(0, 101)] == [0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0]
    # https://oeis.org/A175629
    assert [galois.jacobi_symbol(n, 7) for n in range(0, 87)] == [0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1]
    # https://oeis.org/A011582
    assert [galois.jacobi_symbol(n, 11) for n in range(0, 81)] == [0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1]
    # https://oeis.org/A011583
    assert [galois.jacobi_symbol(n, 13) for n in range(0, 81)] == [0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1]

    # https://oeis.org/A102283
    assert [galois.jacobi_symbol(n, 15) for n in range(0, 76)] == [0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0]
    # https://oeis.org/A322829
    assert [galois.jacobi_symbol(n, 21) for n in range(0, 85)] == [0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0]


def test_kronecker_symbol_exceptions():
    with pytest.raises(TypeError):
        galois.kronecker_symbol(3.0, 7)
    with pytest.raises(TypeError):
        galois.kronecker_symbol(3, 7.0)


def test_kronecker_symbol():
    # https://oeis.org/A102283
    assert [galois.kronecker_symbol(n, 3) for n in range(0, 105)] == [0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1]
    # https://oeis.org/A080891
    assert [galois.kronecker_symbol(n, 5) for n in range(0, 101)] == [0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0,1,-1,-1,1,0]
    # https://oeis.org/A175629
    assert [galois.kronecker_symbol(n, 7) for n in range(0, 87)] == [0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1,-1,1,-1,-1,0,1,1]
    # https://oeis.org/A011582
    assert [galois.kronecker_symbol(n, 11) for n in range(0, 81)] == [0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,-1,1]
    # https://oeis.org/A011583
    assert [galois.kronecker_symbol(n, 13) for n in range(0, 81)] == [0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,0,1,-1]

    # https://oeis.org/A102283
    assert [galois.kronecker_symbol(n, 15) for n in range(0, 76)] == [0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0,1,1,0,1,0,0,-1,1,0,0,-1,0,-1,-1,0]
    # https://oeis.org/A322829
    assert [galois.kronecker_symbol(n, 21) for n in range(0, 85)] == [0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0,1,-1,0,1,1,0,0,-1,0,-1,-1,0,-1,0,0,1,1,0,-1,1,0]

    # https://oeis.org/A289741
    assert [galois.kronecker_symbol(-20, n) for n in range(0, 81)] == [0,1,0,1,0,0,0,1,0,1,0,-1,0,-1,0,0,0,-1,0,-1,0,1,0,1,0,0,0,1,0,1,0,-1,0,-1,0,0,0,-1,0,-1,0,1,0,1,0,0,0,1,0,1,0,-1,0,-1,0,0,0,-1,0,-1,0,1,0,1,0,0,0,1,0,1,0,-1,0,-1,0,0,0,-1,0,-1,0]
    # https://oeis.org/A226162
    assert [galois.kronecker_symbol(-5, n) for n in range(0, 90)] == [0,1,-1,1,1,0,-1,1,-1,1,0,-1,1,-1,-1,0,1,-1,-1,-1,0,1,1,1,-1,0,1,1,1,1,0,-1,-1,-1,1,0,1,-1,1,-1,0,1,-1,1,-1,0,-1,1,1,1,0,-1,-1,-1,-1,0,-1,-1,-1,-1,0,1,1,1,1,0,1,1,-1,1,0,-1,-1,-1,1,0,-1,-1,1,-1,0,1,-1,1,1,0,-1,1,1,1]
    # https://oeis.org/A034947
    assert [galois.kronecker_symbol(-1, n) for n in range(1, 82)] == [1,1,-1,1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,1]
    # https://oeis.org/A091337
    assert [galois.kronecker_symbol(2, n) for n in range(1, 106)] == [1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,-1,0,1,0,1]
    # https://oeis.org/A091338
    assert [galois.kronecker_symbol(3, n) for n in range(1, 103)] == [1,-1,0,1,-1,0,-1,-1,0,1,1,0,1,1,0,1,-1,0,-1,-1,0,-1,1,0,1,-1,0,-1,-1,0,-1,-1,0,1,1,0,1,1,0,1,-1,0,-1,1,0,-1,1,0,1,-1,0,1,-1,0,-1,1,0,1,1,0,1,1,0,1,-1,0,-1,-1,0,-1,1,0,1,-1,0,-1,-1,0,-1,-1,0,1,1,0,1,1,0,-1,-1,0,-1,1,0,-1,1,0,1,-1,0,1,-1,0]
    # https://oeis.org/A322796
    assert [galois.kronecker_symbol(6, n) for n in range(0, 85)] == [0,1,0,0,0,1,0,-1,0,0,0,-1,0,-1,0,0,0,-1,0,1,0,0,0,1,0,1,0,0,0,1,0,-1,0,0,0,-1,0,-1,0,0,0,-1,0,1,0,0,0,1,0,1,0,0,0,1,0,-1,0,0,0,-1,0,-1,0,0,0,-1,0,1,0,0,0,1,0,1,0,0,0,1,0,-1,0,0,0,-1,0]
    # https://oeis.org/A089509
    assert [galois.kronecker_symbol(7, n) for n in range(1, 107)] == [1,1,1,1,-1,1,0,1,1,-1,-1,1,-1,0,-1,1,-1,1,1,-1,0,-1,-1,1,1,-1,1,0,1,-1,1,1,-1,-1,0,1,1,1,-1,-1,-1,0,-1,-1,-1,-1,1,1,0,1,-1,-1,1,1,1,0,1,1,1,-1,-1,1,0,1,1,-1,-1,-1,-1,0,-1,1,-1,1,1,1,0,-1,-1,-1,1,-1,1,0,1,-1,1,-1,-1,-1,0,-1,1,1,-1,1,-1,0,-1,1,-1,-1,1,-1,0,1]


def test_is_cyclic_exceptions():
    with pytest.raises(TypeError):
        galois.is_cyclic(20.0)
    with pytest.raises(ValueError):
        galois.is_cyclic(-1)


def test_is_cyclic():
    assert galois.is_cyclic(2) == True
    assert galois.is_cyclic(4) == True
    assert galois.is_cyclic(2 * 3**3) == True
    assert galois.is_cyclic(5**7) == True
    assert galois.is_cyclic(2**3) == False
    assert galois.is_cyclic(2*5*7*9) == False

    assert all(galois.is_cyclic(n) == (galois.euler_phi(n) == galois.carmichael_lambda(n)) for n in range(1, 100))


###############################################################################
# Integer arithmetic
###############################################################################

def test_isqrt_exceptions():
    with pytest.raises(TypeError):
        galois.isqrt(3.0)
    with pytest.raises(ValueError):
        galois.isqrt(-3)


def test_isqrt():
    """
    Sage:
        N = 20
        n = [randint(0, 1_000_000) for _ in range(N)]
        lut = [(ni, isqrt(ni)) for ni in n]
        print(lut)
    """
    LUT = [(681987, 825), (533875, 730), (743346, 862), (966298, 983), (983657, 991), (208532, 456), (658520, 811), (735666, 857), (155024, 393), (470463, 685), (71083, 266), (706821, 840), (628141, 792), (45582, 213), (460761, 678), (511644, 715), (719018, 847), (596428, 772), (821551, 906), (27234, 165)]
    for item in LUT:
        n, x = item
        assert galois.isqrt(n) == x


def test_iroot_exceptions():
    with pytest.raises(TypeError):
        galois.iroot(9.0, 3)
    with pytest.raises(TypeError):
        galois.iroot(9, 3.0)
    with pytest.raises(ValueError):
        galois.iroot(-9, 3)
    with pytest.raises(ValueError):
        galois.iroot(9, 1)


def test_iroot():
    """
    Sage:
        N = 20
        lut = []
        for _ in range(N):
            n = Integer(randint(0, 1_000_000))
            k = randint(2, 6)
            x = n.nth_root(k, truncate_mode=True)[0]
            lut.append((n, k, x))
        print(lut)
    """
    LUT = [(779174, 4, 29), (867742, 4, 30), (709111, 2, 842), (616365, 6, 9), (615576, 2, 784), (259784, 2, 509), (862570, 2, 928), (553097, 2, 743), (929919, 4, 31), (841722, 6, 9), (658636, 3, 87), (326492, 5, 12), (195217, 4, 21), (969412, 3, 98), (95064, 3, 45), (550943, 3, 81), (171374, 3, 55), (881656, 3, 95), (915960, 6, 9), (810062, 2, 900)]
    for item in LUT:
        n, k, x = item
        assert galois.iroot(n, k) == x

    assert galois.iroot(0, 2) == 0


def test_ilog_exceptions():
    with pytest.raises(TypeError):
        galois.ilog(9.0, 2)
    with pytest.raises(TypeError):
        galois.ilog(9, 2.0)
    with pytest.raises(ValueError):
        galois.ilog(-9, 2)
    with pytest.raises(ValueError):
        galois.ilog(9, 1)


# TODO: Find a way to generate test vectors with Sage
def test_ilog():
    p = galois.mersenne_primes(2000)[-1]
    exponent = galois.ilog(p, 17)
    assert isinstance(exponent, int)
    assert 17**exponent <= p and not 17**(exponent + 1) <= p

    p = galois.mersenne_primes(2000)[-1] - 1
    exponent = galois.ilog(p, 17)
    assert isinstance(exponent, int)
    assert 17**exponent <= p and not 17**(exponent + 1) <= p

    p = galois.mersenne_primes(2000)[-1] + 1
    exponent = galois.ilog(p, 17)
    assert isinstance(exponent, int)
    assert 17**exponent <= p and not 17**(exponent + 1) <= p
