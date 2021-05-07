"""
A pytest module to test polynomial arithmetic implemented with DensePoly, BinaryPoly, and SparsePoly.

We don't need to verify the arithmetic is correct (that was already done in test_arithmetic.py). We
just need to make sure the arithmetic is the same for the different implementations.
"""
import random

import pytest
import numpy as np

import galois


def test_add(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert galois.poly.BinaryPoly._add(a, b) == galois.poly.DensePoly._add(a, b) == galois.poly.SparsePoly._add(a, b)
    else:
        assert galois.poly.DensePoly._add(a, b) == galois.poly.SparsePoly._add(a, b)


def test_subtract(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert galois.poly.BinaryPoly._sub(a, b) == galois.poly.DensePoly._sub(a, b) == galois.poly.SparsePoly._sub(a, b)
    else:
        assert galois.poly.DensePoly._sub(a, b) == galois.poly.SparsePoly._sub(a, b)


def test_multiply(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert galois.poly.BinaryPoly._mul(a, b) == galois.poly.DensePoly._mul(a, b) == galois.poly.SparsePoly._mul(a, b)
    else:
        assert galois.poly.DensePoly._mul(a, b) == galois.poly.SparsePoly._mul(a, b)


def test_divmod(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert galois.poly.BinaryPoly._divmod(a, b) == galois.poly.DensePoly._divmod(a, b) == galois.poly.SparsePoly._divmod(a, b)
    else:
        assert galois.poly.DensePoly._divmod(a, b) == galois.poly.SparsePoly._divmod(a, b)


def test_mod(field):
    a = galois.Poly.Random(random.randint(0, 5))
    b = galois.Poly.Random(random.randint(0, 5))
    if field is galois.GF2:
        assert galois.poly.BinaryPoly._mod(a, b) == galois.poly.DensePoly._mod(a, b) == galois.poly.SparsePoly._mod(a, b)
    else:
        assert galois.poly.DensePoly._mod(a, b) == galois.poly.SparsePoly._mod(a, b)
