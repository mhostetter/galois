"""
A pytest module to test the various algorithms in the galois package.
"""
import math
import random

import pytest
import numpy as np

import galois


def test_gcd():
    a = random.randint(0, 1_000_000)
    b = random.randint(0, 1_000_000)
    gcd, x, y = galois.gcd(a, b)
    assert gcd == math.gcd(a, b)
    assert a*x + b*y == gcd


def test_gcd_exceptions():
    with pytest.raises(TypeError):
        galois.gcd(10.0, 12)
    with pytest.raises(TypeError):
        galois.gcd(10, 12.0)


def test_crt():
    a = [0, 3, 4]
    m = [3, 4, 5]
    x = galois.crt(a, m)
    assert x == 39

    a = [2, 1, 3, 8]
    m = [5, 7, 11, 13]
    x = galois.crt(a, m)
    assert x == 2192


def test_crt_exceptions():
    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 5, 7])
    with pytest.raises(ValueError):
        galois.crt([0, 3, 4], [3, 4, 6])
