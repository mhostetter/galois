"""
A pytest module to test the discrete logarithm implementations.
"""
import random

import pytest

import galois


def test_log_naive():
    """
    (ℤ/14ℤ)* = {1, 3, 5, 9, 11, 13}

    Element:  1, Span: {1}                 , Primitive root: False
    Element:  3, Span: {1, 3, 5, 9, 11, 13}, Primitive root: True
    Element:  5, Span: {1, 3, 5, 9, 11, 13}, Primitive root: True
    Element:  9, Span: {9, 11, 1}          , Primitive root: False
    Element: 11, Span: {9, 11, 1}          , Primitive root: False
    Element: 13, Span: {1, 13}             , Primitive root: False
    """
    n = 14

    beta = 1
    alpha = 1
    x = galois.log_naive(beta, alpha, n)
    assert pow(alpha, x, n) == beta

    beta = 11
    alpha = 3
    x = galois.log_naive(beta, alpha, n)
    assert pow(alpha, x, n) == beta

    beta = 13
    alpha = 5
    x = galois.log_naive(beta, alpha, n)
    assert pow(alpha, x, n) == beta

    beta = 13
    alpha = 9
    with pytest.raises(ValueError):
        x = galois.log_naive(beta, alpha, n)

    beta = 9
    alpha = 11
    x = galois.log_naive(beta, alpha, n)
    assert pow(alpha, x, n) == beta

    beta = 1
    alpha = 13
    x = galois.log_naive(beta, alpha, n)
    assert pow(alpha, x, n) == beta


def test_log_baby_giant_step():
    prime = galois.random_prime(20)
    alpha = galois.primitive_root(prime)
    for _ in range(10):
        beta = random.randint(1, prime-1)
        x = galois.log_baby_giant_step(beta, alpha, prime)
        assert pow(alpha, x, prime) == beta


def test_log_pollard_rho():
    modulus = 383
    alpha = 2
    x = galois.log_pollard_rho(228, alpha, modulus)
    assert x == 110
    assert pow(alpha, x, modulus) == 228


def test_log_pohlig_hellman():
    modulus = 251
    alpha = 71
    x = galois.log_pohlig_hellman(210, alpha, modulus)
    assert x == 197
    assert pow(alpha, x, modulus) == 210


# def test_log_pohlig_hellman_large():
#     modulus = 22708823198678103974314518195029102158525052496759285596453269189798311427475159776411276642277139650833937
#     alpha = 3
#     x = galois.log_pohlig_hellman(123456789, alpha, modulus)
#     assert x == 14779162551733666933424095765296669896168562456605963921868102059346587805144229337814994797820872223240979
#     assert pow(alpha, x, modulus) == 123456789
