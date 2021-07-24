"""
A pytest module to test modular arithmetic functions.
"""
import math
import random

import pytest

import galois


def test_gcd():
    a = random.randint(0, 1_000_000)
    b = random.randint(0, 1_000_000)
    assert galois.gcd(a, b) == math.gcd(a, b)

    with pytest.raises(TypeError):
        galois.gcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.gcd(10, 12.0)


def test_egcd():
    a = random.randint(0, 1_000_000)
    b = random.randint(0, 1_000_000)
    gcd, x, y = galois.egcd(a, b)
    assert gcd == math.gcd(a, b)
    assert a*x + b*y == gcd

    with pytest.raises(TypeError):
        galois.egcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.egcd(10, 12.0)


def test_are_coprime():
    assert galois.are_coprime(3, 4, 5) == True
    assert galois.are_coprime(3, 5, 9, 11) == False
    assert galois.are_coprime(2, 3, 7, 256) == False


def test_crt():
    a = [0, 3, 4]
    m = [3, 4, 5]
    x = galois.crt(a, m)
    assert x == 39

    a = [2, 1, 3, 8]
    m = [5, 7, 11, 13]
    x = galois.crt(a, m)
    assert x == 2192

    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 5, 7])
    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 6])


def test_totatives():
    # https://oeis.org/A000010
    N = list(range(1,70))
    PHI = [1,1,2,2,4,2,6,4,6,4,10,4,12,6,8,8,16,6,18,8,12,10,22,8,20,12,18,12,28,8,30,16,20,16,24,12,36,18,24,16,40,12,42,20,24,22,46,16,42,20,32,24,52,18,40,24,36,28,58,16,60,30,36,32,48,20,66,32,44]
    for n, phi in zip(N, PHI):
        assert len(galois.totatives(n)) == phi

    with pytest.raises(TypeError):
        galois.totatives(20.0)
    with pytest.raises(ValueError):
        galois.totatives(-1)


def test_euler_phi():
    # https://oeis.org/A000010
    N = list(range(1,70))
    PHI = [1,1,2,2,4,2,6,4,6,4,10,4,12,6,8,8,16,6,18,8,12,10,22,8,20,12,18,12,28,8,30,16,20,16,24,12,36,18,24,16,40,12,42,20,24,22,46,16,42,20,32,24,52,18,40,24,36,28,58,16,60,30,36,32,48,20,66,32,44]
    for n, phi in zip(N, PHI):
        assert galois.euler_phi(n) == phi

    with pytest.raises(TypeError):
        galois.euler_phi(20.0)
    with pytest.raises(ValueError):
        galois.euler_phi(-1)


def test_carmichael_lambda():
    # https://oeis.org/A002322
    N = list(range(1,82))
    LAMBDA = [1,1,2,2,4,2,6,2,6,4,10,2,12,6,4,4,16,6,18,4,6,10,22,2,20,12,18,6,28,4,30,8,10,16,12,6,36,18,12,4,40,6,42,10,12,22,46,4,42,20,16,12,52,18,20,6,18,28,58,4,60,30,6,16,12,10,66,16,22,12,70,6,72,36,20,18,30,12,78,4,54]
    for n, lambda_ in zip(N, LAMBDA):
        assert galois.carmichael_lambda(n) == lambda_

    with pytest.raises(TypeError):
        galois.carmichael_lambda(20.0)
    with pytest.raises(ValueError):
        galois.carmichael_lambda(-1)


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


def test_is_cyclic():
    assert galois.is_cyclic(2) == True
    assert galois.is_cyclic(4) == True
    assert galois.is_cyclic(2 * 3**3) == True
    assert galois.is_cyclic(5**7) == True
    assert galois.is_cyclic(2**3) == False
    assert galois.is_cyclic(2*5*7*9) == False

    with pytest.raises(TypeError):
        galois.is_cyclic(20.0)
    with pytest.raises(ValueError):
        galois.is_cyclic(-1)


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
        assert galois.primitive_root(n, reverse=True) == root


def test_smallest_primitive_root_of_primes():
    # https://oeis.org/A001918
    ns = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571]
    roots = [1,2,2,3,2,2,3,2,5,2,3,2,6,3,5,2,2,2,2,7,5,3,2,3,5,2,5,2,6,3,3,2,3,2,2,6,5,2,5,2,2,2,19,5,2,3,2,3,2,6,3,7,7,6,3,5,2,6,5,3,3,2,5,17,10,2,3,10,2,2,3,7,6,2,2,5,2,5,3,21,2,2,7,5,15,2,3,13,2,3,2,13,3,2,7,5,2,3,2,2,2,2,2,3]
    for n, root in zip(ns, roots):
        assert galois.primitive_root(n) == root


def test_largest_primitive_root_of_primes():
    # https://oeis.org/A071894
    ns = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281]
    roots = [1,2,3,5,8,11,14,15,21,27,24,35,35,34,45,51,56,59,63,69,68,77,80,86,92,99,101,104,103,110,118,128,134,135,147,146,152,159,165,171,176,179,189,188,195,197,207,214,224,223,230,237,234,248,254,261,267,269,272]
    for n, root in zip(ns, roots):
        assert galois.primitive_root(n, reverse=True) == root


def test_primitive_root_exceptions():
    with pytest.raises(TypeError):
        galois.primitive_root(20.0)
    with pytest.raises(TypeError):
        galois.primitive_root(20, start=1.0)
    with pytest.raises(TypeError):
        galois.primitive_root(20, stop=20.0)
    with pytest.raises(TypeError):
        galois.primitive_root(20, reverse=1)

    with pytest.raises(ValueError):
        galois.primitive_root(0)
    with pytest.raises(ValueError):
        galois.primitive_root(-2)
    with pytest.raises(ValueError):
        galois.primitive_root(7, start=0)
    with pytest.raises(ValueError):
        galois.primitive_root(7, start=7)
    with pytest.raises(ValueError):
        galois.primitive_root(7, stop=1)
    with pytest.raises(ValueError):
        galois.primitive_root(7, start=6, stop=6)


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
    assert galois.primitive_roots(16) == []
    assert galois.primitive_roots(17) == [3,5,6,7,10,11,12,14]
    assert galois.primitive_roots(18) == [5,11]
    assert galois.primitive_roots(19) == [2,3,10,13,14,15]
    assert galois.primitive_roots(20) == []
    assert galois.primitive_roots(21) == []
    assert galois.primitive_roots(22) == [7,13,17,19]
    assert galois.primitive_roots(23) == [5,7,10,11,14,15,17,19,20,21]
    assert galois.primitive_roots(24) == []
    assert galois.primitive_roots(25) == [2,3,8,12,13,17,22,23]
    assert galois.primitive_roots(26) == [7,11,15,19]
    assert galois.primitive_roots(27) == [2,5,11,14,20,23]
    assert galois.primitive_roots(28) == []
    assert galois.primitive_roots(29) == [2,3,8,10,11,14,15,18,19,21,26,27]
    assert galois.primitive_roots(30) == []


def test_number_of_primitive_roots():
    # https://oeis.org/A046144
    ns = list(range(1,92))
    num_roots = [1,1,1,1,2,1,2,0,2,2,4,0,4,2,0,0,8,2,6,0,0,4,10,0,8,4,6,0,12,0,8,0,0,8,0,0,12,6,0,0,16,0,12,0,0,10,22,0,12,8,0,0,24,6,0,0,0,12,28,0,16,8,0,0,0,0,20,0,0,0,24,0,24,12,0,0,0,0,24,0,18,16,40,0,0,12,0,0,40,0,0]
    for n, num in zip(ns, num_roots):
        assert len(galois.primitive_roots(n)) == num


def test_number_of_primitive_roots_of_primes():
    # https://oeis.org/A008330
    ns = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353]
    num_roots = [1,1,2,2,4,4,8,6,10,12,8,12,16,12,22,24,28,16,20,24,24,24,40,40,32,40,32,52,36,48,36,48,64,44,72,40,48,54,82,84,88,48,72,64,84,60,48,72,112,72,112,96,64,100,128,130,132,72,88,96,92,144,96,120,96,156,80,96,172,112]
    for n, num in zip(ns, num_roots):
        assert len(galois.primitive_roots(n)) == num


def test_sum_primitive_roots_of_primes():
    # https://oeis.org/A088144
    ns = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199]
    sums = [1,2,5,8,23,26,68,57,139,174,123,222,328,257,612,636,886,488,669,1064,876,1105,1744,1780,1552,2020,1853,2890,1962,2712,2413,3536,4384,3335,5364,3322,3768,4564,7683,7266,8235,4344,8021,6176,8274]
    for n, s in zip(ns, sums):
        assert sum(galois.primitive_roots(n)) == s


@pytest.mark.parametrize("n", [2, 4, 7**2, 2*257])
def test_primitive_roots_are_generators(n):
    n = int(n)
    congruences = [a for a in range(1, n) if math.gcd(n, a) == 1]
    phi = galois.euler_phi(n)
    assert len(congruences) == phi

    roots = galois.primitive_roots(n)
    for root in roots:
        elements = [pow(root, i, n) for i in range(1, n)]
        assert set(congruences) == set(elements)

    assert len(roots) == galois.euler_phi(phi)


def test_primitive_roots_exceptions():
    with pytest.raises(TypeError):
        galois.primitive_roots(20.0)
    with pytest.raises(TypeError):
        galois.primitive_roots(20, start=1.0)
    with pytest.raises(TypeError):
        galois.primitive_roots(20, stop=20.0)
    with pytest.raises(TypeError):
        galois.primitive_roots(20, reverse=1)

    with pytest.raises(ValueError):
        galois.primitive_roots(0)
    with pytest.raises(ValueError):
        galois.primitive_roots(-2)
    with pytest.raises(ValueError):
        galois.primitive_roots(7, start=0)
    with pytest.raises(ValueError):
        galois.primitive_roots(7, start=7)
    with pytest.raises(ValueError):
        galois.primitive_roots(7, stop=1)
    with pytest.raises(ValueError):
        galois.primitive_roots(7, start=6, stop=6)


def test_is_primitive_root():
    n = 7
    roots = [3, 5]
    for g in range(1, 7):
        if g in roots:
            assert galois.is_primitive_root(g, n)
        else:
            assert not galois.is_primitive_root(g, n)


def test_is_primitive_root_exceptions():
    with pytest.raises(TypeError):
        galois.is_primitive_root(3.0, 13)
    with pytest.raises(TypeError):
        galois.is_primitive_root(3, 13.0)
    with pytest.raises(ValueError):
        galois.is_primitive_root(3, 0)
    with pytest.raises(ValueError):
        galois.is_primitive_root(15, 13)
