"""
A pytest module to test ufunc methods on FieldArrays.
"""

import numpy as np
import pytest

from .conftest import randint

# TODO: Test using "out" keyword argument


class TestReduce:
    """
    Tests the `reduce` method.
    """

    def test_add(self, field):
        a = field.Random(10)
        b = np.add.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth + ai
        assert b == b_truth

    def test_negative(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.negative.reduce(a)

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

    def test_reciprocal(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.reciprocal.reduce(a)

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

    def test_power(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.power.reduce(a)

    def test_square(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.square.reduce(a)

    def test_log(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.log.reduce(a)


class TestAccumulate:
    """
    Tests the `accumulate` method.
    """

    def test_add(self, field):
        a = field.Random(10)
        b = np.add.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i - 1] + a[i]
        assert np.array_equal(b, b_truth)

    def test_negative(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.negative.accumulate(a)

    def test_subtract(self, field):
        a = field.Random(10)
        b = np.subtract.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i - 1] - a[i]
        assert np.array_equal(b, b_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        b = np.multiply.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i - 1] * a[i]
        assert np.array_equal(b, b_truth)

    def test_reciprocal(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.reciprocal.accumulate(a)

    def test_divide(self, field):
        a = field.Random(10, low=1)
        b = np.true_divide.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i - 1] / a[i]
        assert np.array_equal(b, b_truth)

        a = field.Random(10, low=1)
        b = np.floor_divide.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i - 1] // a[i]
        assert np.array_equal(b, b_truth)

    def test_power(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.power.accumulate(a)

    def test_square(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.square.accumulate(a)

    def test_log(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.log.accumulate(a)


class TestReduceAt:
    """
    Tests the `reduceat` method.
    """

    def test_add(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        b = np.add.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.add.reduce(a[idxs[i] :])
            else:
                b_truth[i] = np.add.reduce(a[idxs[i] : idxs[i + 1]])
        assert np.array_equal(b, b_truth)

    def test_negative(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        with pytest.raises(ValueError):
            np.negative.reduceat(a, idxs)

    def test_subtract(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        b = np.subtract.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.subtract.reduce(a[idxs[i] :])
            else:
                b_truth[i] = np.subtract.reduce(a[idxs[i] : idxs[i + 1]])
        assert np.array_equal(b, b_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        b = np.multiply.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.multiply.reduce(a[idxs[i] :])
            else:
                b_truth[i] = np.multiply.reduce(a[idxs[i] : idxs[i + 1]])
        assert np.array_equal(b, b_truth)

    def test_reciprocal(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        with pytest.raises(ValueError):
            np.reciprocal.reduceat(a, idxs)

    def test_divide(self, field):
        a = field.Random(10, low=1)
        idxs = [1, 4, 5, 8]
        b = np.true_divide.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.true_divide.reduce(a[idxs[i] :])
            else:
                b_truth[i] = np.true_divide.reduce(a[idxs[i] : idxs[i + 1]])
        assert np.array_equal(b, b_truth)

        a = field.Random(10, low=1)
        idxs = [1, 4, 5, 8]
        b = np.floor_divide.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.floor_divide.reduce(a[idxs[i] :])
            else:
                b_truth[i] = np.floor_divide.reduce(a[idxs[i] : idxs[i + 1]])
        assert np.array_equal(b, b_truth)

    def test_power(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        with pytest.raises(ValueError):
            np.power.reduceat(a, idxs)

    def test_square(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        with pytest.raises(ValueError):
            np.square.reduceat(a, idxs)

    def test_log(self, field):
        a = field.Random(10)
        idxs = [1, 4, 5, 8]
        with pytest.raises(ValueError):
            np.log.reduceat(a, idxs)


class TestOuter:
    """
    Tests the `outer` method.
    """

    def test_add(self, field):
        a = field.Random(10)
        b = field.Random(12)
        c = np.add.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i, j] = a[i] + b[j]
        assert np.array_equal(c, c_truth)

    def test_negative(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.negative.outer(a)

    def test_subtract(self, field):
        a = field.Random(10)
        b = field.Random(12)
        c = np.subtract.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i, j] = a[i] - b[j]
        assert np.array_equal(c, c_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        b = field.Random(12)
        # b = randint(0, field.order, 12, field.dtypes[-1])  # Why do this? It's scalar multiplication....
        c = np.multiply.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i, j] = a[i] * b[j]
        assert np.array_equal(c, c_truth)

    def test_reciprocal(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.reciprocal.outer(a)

    def test_divide(self, field):
        a = field.Random(10)
        b = field.Random(12, low=1)
        c = np.true_divide.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i, j] = a[i] / b[j]
        assert np.array_equal(c, c_truth)

        a = field.Random(10)
        b = field.Random(12, low=1)
        c = np.floor_divide.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i, j] = a[i] // b[j]
        assert np.array_equal(c, c_truth)

    def test_power(self, field):
        a = field.Random(10)
        b = randint(1, field.order, 12, field.dtypes[-1])
        c = np.power.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i, j] = a[i] ** b[j]
        assert np.array_equal(c, c_truth)

    def test_square(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.square.outer(a)

    def test_log(self, field):
        a = field.Random(10)
        with pytest.raises(ValueError):
            np.log.outer(a)


class TestAt:
    """
    Tests the `at` method.
    """

    def test_add(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.add.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] + b
        assert np.array_equal(a, a_truth)

    def test_negative(self, field):
        a = field.Random(10)
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.negative.at(a, idxs)
        for i in idxs:
            a_truth[i] = np.negative(a_truth[i])
        assert np.array_equal(a, a_truth)

    def test_subtract(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.subtract.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] - b
        assert np.array_equal(a, a_truth)

    def test_multiply(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.multiply.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] * b
        assert np.array_equal(a, a_truth)

    def test_reciprocal(self, field):
        a = field.Random(10, low=1)
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.reciprocal.at(a, idxs)
        for i in idxs:
            a_truth[i] = np.reciprocal(a_truth[i])
        assert np.array_equal(a, a_truth)

    def test_divide(self, field):
        a = field.Random(10)
        b = field.Random(low=1)
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.true_divide.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] / b
        assert np.array_equal(a, a_truth)

        a = field.Random(10)
        b = field.Random(low=1)
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.floor_divide.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] // b
        assert np.array_equal(a, a_truth)

    def test_power(self, field):
        a = field.Random(10)
        b = randint(1, field.order, 1, field.dtypes[-1])
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.power.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] ** b
        assert np.array_equal(a, a_truth)

    def test_square(self, field):
        a = field.Random(10)
        idxs = [0, 1, 1, 4, 8]
        a_truth = field(a)  # Ensure a copy happens
        np.square.at(a, idxs)
        for i in idxs:
            a_truth[i] = np.square(a_truth[i])
        assert np.array_equal(a, a_truth)
