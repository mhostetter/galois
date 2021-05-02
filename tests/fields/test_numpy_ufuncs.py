"""
A pytest module to test ufunc methods on Galois field arrays.
"""
import numpy as np
import pytest

import galois

from ..helper import randint


# TODO: Test using "out" keyword argument

class TestReduce:
    def test_add(self, field):
        a = field.Random(10)
        b = np.add.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth + ai
        assert b == b_truth

    def test_subtract(self, field):
        a = field.Random(10)
        b = np.subtract.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth - ai
        assert b == b_truth

    def test_multiply(self, field):
        a = field.Random(10)
        b = np.multiply.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth * ai
        assert b == b_truth

    def test_divide(self, field):
        a = field.Random(10, low=1)
        b = np.true_divide.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth / ai
        assert b == b_truth

        a = field.Random(10, low=1)
        b = np.floor_divide.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth // ai
        assert b == b_truth

    # TODO: Revisist if this is valid
    # def test_power(self, field):
    #     a = field.Random(10)
    #     b = np.power.reduce(a)
    #     b_truth = a[0]
    #     for ai in a[1:]:
    #         b_truth = b_truth ** ai
    #     assert b == b_truth


class TestAccumulate:
    def test_add(self, field):
        a = field.Random(10)
        b = np.add.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] + a[i]
        assert np.array_equal(b, b_truth)

    def test_subtract(self, field):
        a = field.Random(10)
        b = np.subtract.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] - a[i]
        assert np.array_equal(b, b_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        b = np.multiply.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] * a[i]
        assert np.array_equal(b, b_truth)

    def test_divide(self, field):
        a = field.Random(10, low=1)
        b = np.true_divide.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] / a[i]
        assert np.array_equal(b, b_truth)

        a = field.Random(10, low=1)
        b = np.floor_divide.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] // a[i]
        assert np.array_equal(b, b_truth)

    # TODO: Revisist if this is valid
    # def test_power(self, field):
    #     a = field.Random(10)
    #     b = np.power.accumulate(a)
    #     b_truth = field.Zeros(10)
    #     b_truth[0] = a[0]
    #     for i in range(1, 10):
    #         b_truth[i] = b_truth[i-1] ** a[i]
    #     assert np.array_equal(b, b_truth)


class TestReduceAt:
    def test_add(self, field):
        a = field.Random(10)
        idxs = [1,4,5,8]
        b = np.add.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.add.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.add.reduce(a[idxs[i]:idxs[i+1]])
        assert np.array_equal(b, b_truth)

    def test_subtract(self, field):
        a = field.Random(10)
        idxs = [1,4,5,8]
        b = np.subtract.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.subtract.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.subtract.reduce(a[idxs[i]:idxs[i+1]])
        assert np.array_equal(b, b_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        idxs = [1,4,5,8]
        b = np.multiply.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.multiply.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.multiply.reduce(a[idxs[i]:idxs[i+1]])
        assert np.array_equal(b, b_truth)

    def test_divide(self, field):
        a = field.Random(10, low=1)
        idxs = [1,4,5,8]
        b = np.true_divide.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.true_divide.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.true_divide.reduce(a[idxs[i]:idxs[i+1]])
        assert np.array_equal(b, b_truth)

        a = field.Random(10, low=1)
        idxs = [1,4,5,8]
        b = np.floor_divide.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.floor_divide.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.floor_divide.reduce(a[idxs[i]:idxs[i+1]])
        assert np.array_equal(b, b_truth)

    # TODO: Revisist if this is valid
    # def test_power(self, field):
    #     a = field.Random(10)
    #     idxs = [1,4,5,8]
    #     b = np.power.reduceat(a, idxs)
    #     b_truth = field.Zeros(len(idxs))
    #     for i in range(len(idxs)):
    #         if i == len(idxs) - 1:
    #             b_truth[i] = np.power.reduce(a[idxs[i]:])
    #         else:
    #             b_truth[i] = np.power.reduce(a[idxs[i]:idxs[i+1]])
    #     assert np.array_equal(b, b_truth)


class TestOuter:
    def test_add(self, field):
        a = field.Random(10)
        b = field.Random(12)
        c = np.add.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] + b[j]
        assert np.array_equal(c, c_truth)

    def test_subtract(self, field):
        a = field.Random(10)
        b = field.Random(12)
        c = np.subtract.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] - b[j]
        assert np.array_equal(c, c_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        b = randint(0, field.order, 12, field.dtypes[-1])
        c = np.multiply.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] * b[j]
        assert np.array_equal(c, c_truth)

    def test_divide(self, field):
        a = field.Random(10)
        b = field.Random(12, low=1)
        c = np.true_divide.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] / b[j]
        assert np.array_equal(c, c_truth)

        a = field.Random(10)
        b = field.Random(12, low=1)
        c = np.floor_divide.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] // b[j]
        assert np.array_equal(c, c_truth)

    def test_power(self, field):
        a = field.Random(10)
        b = randint(1, field.order, 12, field.dtypes[-1])
        c = np.power.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] ** b[j]
        assert np.array_equal(c, c_truth)


class TestAt:
    def test_add(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.add.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] + b
        assert np.array_equal(a, a_truth)

    def test_subtract(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.subtract.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] - b
        assert np.array_equal(a, a_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.multiply.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] * b
        assert np.array_equal(a, a_truth)

    def test_divide(self, field):
        a = field.Random(10)
        b = field.Random(low=1)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.true_divide.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] / b
        assert np.array_equal(a, a_truth)

        a = field.Random(10)
        b = field.Random(low=1)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.floor_divide.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] // b
        assert np.array_equal(a, a_truth)

    def test_negative(self, field):
        a = field.Random(10)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.negative.at(a, idxs)
        for i in idxs:
            a_truth[i] = -a_truth[i]
        assert np.array_equal(a, a_truth)

    def test_power(self, field):
        a = field.Random(10)
        b = randint(1, field.order, 1, field.dtypes[-1])
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.power.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] ** b
        assert np.array_equal(a, a_truth)

    def test_log(self, field):
        if field.order > 2**16:  # TODO: Skip slow log() for very large fields
            return
        a = field.Random(10, low=1)
        idxs = [0,1,4,8]  # Dont test index=1 twice like other tests because in GF(2) log(1)=0 and then log(0)=error
        a_truth = field(a)  # Ensure a copy happens
        np.log.at(a, idxs)
        for i in idxs:
            a_truth[i] = np.log(a_truth[i])
        assert np.array_equal(a, a_truth)
