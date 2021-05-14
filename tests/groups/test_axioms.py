"""
A pytest module to test the group axioms on additive and multiplicative groups.
"""
import pytest
import numpy as np

import galois


class TestAdditive:
    def test_closure(self, additive_group):
        G = additive_group
        a = G.Random(10)
        b = G.Random(10)
        c = a + b
        assert np.all(0 <= c) and np.all(c < G.modulus)

    def test_associativity(self, additive_group):
        G = additive_group
        a = G.Random(10)
        b = G.Random(10)
        c = G.Random(10)
        assert np.array_equal((a + b) + c, a + (b + c))

    def test_identity(self, additive_group):
        G = additive_group
        a = G.Random(10)
        e = G.identity
        assert np.all(a + e == a)
        assert np.all(e + a == a)

    def test_inverse(self, additive_group):
        G = additive_group
        a = G.Random(10)
        b = np.negative(a)
        e = G.identity
        assert np.all(a + b == e)
        assert np.all(b + a == e)

    def test_commutativity(self, additive_group):
        G = additive_group
        a = G.Random(10)
        b = G.Random(10)
        assert np.array_equal(a + b, b + a)


class TestMultiplicative:
    def test_closure(self, multiplicative_group):
        G = multiplicative_group
        a = G.Random(10)
        b = G.Random(10)
        c = a * b
        assert np.all(0 <= c) and np.all(c < G.modulus)

    def test_associativity(self, multiplicative_group):
        G = multiplicative_group
        a = G.Random(10)
        b = G.Random(10)
        c = G.Random(10)
        assert np.array_equal((a * b) * c, a * (b * c))

    def test_identity(self, multiplicative_group):
        G = multiplicative_group
        a = G.Random(10)
        e = G.identity
        assert np.all(a * e == a)
        assert np.all(e * a == a)

    def test_inverse(self, multiplicative_group):
        G = multiplicative_group
        a = G.Random(10)
        b = np.reciprocal(a)
        e = G.identity
        assert np.all(a * b == e)
        assert np.all(b * a == e)

    def test_commutativity(self, multiplicative_group):
        G = multiplicative_group
        a = G.Random(10)
        b = G.Random(10)
        assert np.array_equal(a * b, b * a)
