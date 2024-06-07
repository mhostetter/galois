"""
A pytest module to test proper FieldArray broadcasting.
"""

import numpy as np

from .conftest import randint

# NOTE: We don't need to verify the arithmetic is correct here, that was done in test_field_arithmetic.py


def test_scalar_scalar(field):
    shape1 = ()
    shape2 = ()
    shape_result = ()
    check_results(field, shape1, shape2, shape_result)


def test_array_scalar(field):
    shape1 = (2, 2)
    shape2 = ()
    shape_result = (2, 2)
    check_results(field, shape1, shape2, shape_result)


def test_scalar_array(field):
    shape1 = ()
    shape2 = (2, 2)
    shape_result = (2, 2)
    check_results(field, shape1, shape2, shape_result)


def test_array_array(field):
    shape1 = (2, 4)
    shape2 = (4,)
    shape_result = (2, 4)
    check_results(field, shape1, shape2, shape_result)


def check_results(field, shape1, shape2, shape_result):
    a = field.Random(shape1)
    b = field.Random(shape2, low=1)
    c = randint(0, field.order, shape2, field.dtypes[-1])
    d = 2 * np.ones(shape2, field.dtypes[-1])

    # Test np.add ufunc
    z = a + b
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.subtract ufunc
    z = a - b
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.multiply ufunc
    z = a * b
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.true_divide ufunc
    z = a / b
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.floor_divide ufunc
    z = a // b
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.multiply ufunc (multiple addition)
    z = a * c
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.power ufunc
    z = a**c
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.square ufunc
    z = a**d
    assert type(z) is field
    assert z.shape == shape_result

    # Test np.negative ufunc
    z = -a
    assert type(z) is field
    assert z.shape == a.shape

    # Test np.log ufunc
    if field.order <= 2**16:  # TODO: Skip slow log() for very large fields
        z = np.log(b)
        assert z.shape == b.shape
