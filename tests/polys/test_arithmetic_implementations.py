"""
A pytest module to test polynomial arithmetic implemented with DensePoly, BinaryPoly, and SparsePoly.

We don't need to verify the arithmetic is correct (that was already done in test_arithmetic.py). We
just need to make sure the arithmetic is the same for the different implementations.
"""
import random

import galois
from galois._fields._main import DensePoly, BinaryPoly, SparsePoly


def test_add(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert BinaryPoly._add(a, b) == DensePoly._add(a, b) == SparsePoly._add(a, b)
    else:
        assert DensePoly._add(a, b) == SparsePoly._add(a, b)


def test_subtract(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert BinaryPoly._sub(a, b) == DensePoly._sub(a, b) == SparsePoly._sub(a, b)
    else:
        assert DensePoly._sub(a, b) == SparsePoly._sub(a, b)


def test_multiply(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert BinaryPoly._mul(a, b) == DensePoly._mul(a, b) == SparsePoly._mul(a, b)
    else:
        assert DensePoly._mul(a, b) == SparsePoly._mul(a, b)


def test_divmod(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert BinaryPoly._divmod(a, b) == DensePoly._divmod(a, b) == SparsePoly._divmod(a, b)
    else:
        assert DensePoly._divmod(a, b) == SparsePoly._divmod(a, b)


def test_mod(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert BinaryPoly._mod(a, b) == DensePoly._mod(a, b) == SparsePoly._mod(a, b)
    else:
        assert DensePoly._mod(a, b) == SparsePoly._mod(a, b)
