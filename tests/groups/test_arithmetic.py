"""
A pytest module to test the accuracy of finite group array arithmetic.
"""
import random

import pytest
import numpy as np

import galois

from ..helper import randint, array_equal


class TestAdditive:
    def test_add(self, additive_group):
        G = additive_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)
        y = G.Random(10, dtype=dtype)

        z = x + y
        Z = (np.array(x, dtype=object) + np.array(y, dtype=object)) % G.modulus
        assert array_equal(z, Z)
        assert type(z) is G
        assert z.dtype == dtype

    def test_additive_inverse(self, additive_group):
        G = additive_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)

        z = x + np.negative(x)
        assert np.all(z == 0)
        assert type(z) is G
        assert z.dtype == dtype

    def test_power(self, additive_group):
        G = additive_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)
        y = randint(-G.modulus, G.modulus, 10, dtype)

        z = x ** y
        Z = (np.array(x, dtype=object) * np.array(y, dtype=object)) % G.modulus
        assert array_equal(z, Z)
        assert type(z) is G
        assert z.dtype == dtype

    def test_square(self, additive_group):
        G = additive_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)
        y = 2*np.ones(10, dtype=dtype)

        z = x ** y
        Z = (np.array(x, dtype=object) * 2) % G.modulus
        assert array_equal(z, Z)
        assert type(z) is G
        assert z.dtype == dtype


class TestMultiplicative:
    def test_multiply(self, multiplicative_group):
        G = multiplicative_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)
        y = G.Random(10, dtype=dtype)

        z = x * y
        Z = (np.array(x, dtype=object) * np.array(y, dtype=object)) % G.modulus
        assert array_equal(z, Z)
        assert type(z) is G
        assert z.dtype == dtype

    def test_multiplicative_inverse(self, multiplicative_group):
        G = multiplicative_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)

        z = x * np.reciprocal(x)
        assert np.all(z == 1)
        assert type(z) is G
        assert z.dtype == dtype

    def test_power(self, multiplicative_group):
        G = multiplicative_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)
        y = randint(0, G.modulus, 10, dtype)

        z = x ** y
        Z = [pow(int(x[i]), int(y[i]), G.modulus) for i in range(x.size)]
        print(x)
        print(y)
        print(z)
        print(Z)
        assert array_equal(z, Z)
        assert type(z) is G
        assert z.dtype == dtype

    def test_square(self, multiplicative_group):
        G = multiplicative_group
        dtype = random.choice(G.dtypes)
        x = G.Random(10, dtype=dtype)
        y = 2*np.ones(10, dtype=dtype)

        z = x ** y
        Z = [pow(int(x[i]), int(y[i]), G.modulus) for i in range(x.size)]
        assert array_equal(z, Z)
        assert type(z) is G
        assert z.dtype == dtype


# def test_power(power):
#     G, X, Y, Z = power["G"], power["X"], power["Y"], power["Z"]
#     dtype = random.choice(G.dtypes)
#     x = X.astype(dtype)
#     y = Y  # Don't convert this, it's not a field element

#     z = x ** y
#     assert np.array_equal(z, Z)
#     assert type(z) is G
#     assert z.dtype == dtype


# def test_power_zero_to_zero(power):
#     G = power["G"]
#     dtype = random.choice(G.dtypes)
#     x = G.Zeros(10, dtype=dtype)
#     y = np.zeros(10, G.dtypes[-1])
#     z = x ** y
#     Z = np.ones(10, G.dtypes[-1])
#     assert np.array_equal(z, Z)
#     assert type(z) is G
#     assert z.dtype == dtype


# def test_power_zero_to_positive_integer(power):
#     G = power["G"]
#     dtype = random.choice(G.dtypes)
#     x = G.Zeros(10, dtype=dtype)
#     y = randint(1, 2*G.order, 10, G.dtypes[-1])
#     z = x ** y
#     Z = np.zeros(10, G.dtypes[-1])
#     assert np.array_equal(z, Z)
#     assert type(z) is G
#     assert z.dtype == dtype


# def test_square(power):
#     G, X, Y, Z = power["G"], power["X"], power["Y"], power["Z"]
#     dtype = random.choice(G.dtypes)
#     x = X.astype(dtype)
#     y = Y  # Don't convert this, it's not a field element

#     # Not guaranteed to have y=2 for "sparse" LUTs
#     if np.where(Y == 2)[1].size > 0:
#         j = np.where(y == 2)[1][0]  # Index of Y where y=2
#         x = x[:,j]
#         z = x ** 2
#         assert np.array_equal(z, Z[:,j])
#         assert type(z) is G
#         assert z.dtype == dtype


# def test_log(log):
#     G, X, Z = log["G"], log["X"], log["Z"]
#     dtype = random.choice(G.dtypes)
#     if G.order > 2**16:  # TODO: Skip slow log() for very large fields
#         return
#     x = X.astype(dtype)
#     z = np.log(x)
#     assert np.array_equal(z, Z)
