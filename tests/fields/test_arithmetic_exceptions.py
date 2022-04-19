"""
A pytest module to test exception raising for invalid FieldArray arithmetic.
"""
import pytest
import numpy as np

import galois

from ..helper import randint


def test_add_int_scalar(field):
    x = field.Random(10)
    y = int(randint(0, field.order, 1, field.dtypes[-1]))
    with pytest.raises(TypeError):
        z = x + y
    with pytest.raises(TypeError):
        z = y + x


def test_add_int_array(field):
    x = field.Random(10)
    y = randint(0, field.order, 10, field.dtypes[-1])
    with pytest.raises(TypeError):
        z = x + y
    with pytest.raises(TypeError):
        z = y + x


def test_right_add_int_scalar(field):
    x = field.Random(10)
    y = int(randint(0, field.order, 1, field.dtypes[-1]))
    with pytest.raises(TypeError):
        x += y
    with pytest.raises(TypeError):
        y += x


def test_right_add_int_array(field):
    x = field.Random(10)
    y = randint(0, field.order, 10, field.dtypes[-1])
    with pytest.raises(TypeError):
        x += y
    with pytest.raises(TypeError):
        y += x


def test_subtract_int_scalar(field):
    x = field.Random(10)
    y = int(randint(0, field.order, 1, field.dtypes[-1]))
    with pytest.raises(TypeError):
        z = x - y
    with pytest.raises(TypeError):
        z = y - x


def test_subtract_int_array(field):
    x = field.Random(10)
    y = randint(0, field.order, 10, field.dtypes[-1])
    with pytest.raises(TypeError):
        z = x - y
    with pytest.raises(TypeError):
        z = y - x


def test_right_subtract_int_scalar(field):
    x = field.Random(10)
    y = int(randint(0, field.order, 1, field.dtypes[-1]))
    with pytest.raises(TypeError):
        x -= y
    with pytest.raises(TypeError):
        y -= x


def test_right_subtract_int_array(field):
    x = field.Random(10)
    y = randint(0, field.order, 10, field.dtypes[-1])
    with pytest.raises(TypeError):
        x -= y
    with pytest.raises(TypeError):
        y -= x


# NOTE: Don't test multiply with integer because that is a valid operation, namely "multiple addition"


def test_divide_int_scalar(field):
    x = field.Random(10, low=1)
    y = int(randint(1, field.order, 1, field.dtypes[-1]))
    with pytest.raises(TypeError):
        z = x / y
    with pytest.raises(TypeError):
        z = x // y
    with pytest.raises(TypeError):
        z = y / x
    with pytest.raises(TypeError):
        z = y // x


def test_divide_int_array(field):
    x = field.Random(10, low=1)
    y = randint(1, field.order, 10, field.dtypes[-1])
    with pytest.raises(TypeError):
        z = x / y
    with pytest.raises(TypeError):
        z = x // y
    with pytest.raises(TypeError):
        z = y / x
    with pytest.raises(TypeError):
        z = y // x


def test_right_divide_int_scalar(field):
    x = field.Random(10, low=1)
    y = int(randint(1, field.order, 1, field.dtypes[-1]))
    with pytest.raises(TypeError):
        x /= y
    with pytest.raises(TypeError):
        x //= y
    with pytest.raises(TypeError):
        y /= x
    with pytest.raises(TypeError):
        y //= x


def test_right_divide_int_array(field):
    x = field.Random(10, low=1)
    y = randint(1, field.order, 10, field.dtypes[-1])
    with pytest.raises(TypeError):
        x /= y
    with pytest.raises(TypeError):
        x //= y
    with pytest.raises(TypeError):
        y /= x
    with pytest.raises(TypeError):
        y //= x


def test_divide_by_zero(field):
    x = field.Random(10)
    with pytest.raises(ZeroDivisionError):
        y = field(0)
        z = x / y
    with pytest.raises(ZeroDivisionError):
        y = field.Random(10)
        y[0] = 0  # Ensure one value is zero
        z = x / y


def test_multiplicative_inverse_of_zero(field):
    x = field.Random(10)
    x[0] = 0  # Ensure one value is zero
    with pytest.raises(ZeroDivisionError):
        z = x ** -1


# NOTE: Don't test power to integer because that's valid


def test_zero_to_negative_power(field):
    x = field.Random(10)
    x[0] = 0  # Ensure one value is zero
    with pytest.raises(ZeroDivisionError):
        y = -3
        z = x ** y
    with pytest.raises(ZeroDivisionError):
        y = -3*np.ones(x.size, field.dtypes[-1])
        z = x ** y


def test_log_of_zero(field):
    with pytest.raises(ArithmeticError):
        x = field(0)
        z = np.log(x)
    with pytest.raises(ArithmeticError):
        x = field.Zeros(10)
        z = np.log(x)
