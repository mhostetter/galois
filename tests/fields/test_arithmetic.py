"""
A pytest module to test the accuracy of Galois field array arithmetic.
"""
import numpy as np
import pytest

import galois

from ..helper import ALL_DTYPES, randint


# TODO: Add scalar arithmetic and array/scalar and radd, etc


class TestAccuracy:
    def test_add(self, add):
        GF, X, Y, Z = add["GF"], add["X"], add["Y"], add["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y.astype(dtype)

            z = x + y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_subtract(self, subtract):
        GF, X, Y, Z = subtract["GF"], subtract["X"], subtract["Y"], subtract["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y.astype(dtype)

            z = x - y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_multiply(self, multiply):
        GF, X, Y, Z = multiply["GF"], multiply["X"], multiply["Y"], multiply["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y.astype(dtype)

            z = x * y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_divide(self, divide):
        GF, X, Y, Z = divide["GF"], divide["X"], divide["Y"], divide["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y.astype(dtype)

            z = x / y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

            z = x // y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_additive_inverse(self, additive_inverse):
        GF, X, Z = additive_inverse["GF"], additive_inverse["X"], additive_inverse["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)

            z = -x
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_multiplicative_inverse(self, multiplicative_inverse):
        GF, X, Z = multiplicative_inverse["GF"], multiplicative_inverse["X"], multiplicative_inverse["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)

            z = GF(1, dtype=dtype) / x  # Need dtype of "1" to be as large as x for `z.dtype == dtype`
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

            z = GF(1, dtype=dtype) // x  # Need dtype of "1" to be as large as x for `z.dtype == dtype`
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

            z = x ** -1
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_multiple_add(self, multiple_add):
        GF, X, Y, Z = multiple_add["GF"], multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y  # Don't convert this, it's not a field element

            z = x * y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_power(self, power):
        GF, X, Y, Z = power["GF"], power["X"], power["Y"], power["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y  # Don't convert this, it's not a field element

            z = x ** y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_power_zero_to_zero(self, field):
        for dtype in field.dtypes:
            x = field.Zeros(10, dtype=dtype)
            y = np.zeros(10, field.dtypes[-1])
            z = x ** y
            Z = np.ones(10, field.dtypes[-1])
            assert np.all(z == Z)
            assert type(z) is field
            assert z.dtype == dtype

    def test_power_zero_to_positive_integer(self, field):
        for dtype in field.dtypes:
            x = field.Zeros(10, dtype=dtype)
            y = randint(1, 2*field.order, 10, field.dtypes[-1])
            z = x ** y
            Z = np.zeros(10, field.dtypes[-1])
            assert np.all(z == Z)
            assert type(z) is field
            assert z.dtype == dtype

    def test_square(self, power):
        GF, X, Y, Z = power["GF"], power["X"], power["Y"], power["Z"]
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y  # Don't convert this, it's not a field element

            # Not guaranteed to have y=2 for "sparse" LUTs
            if np.where(Y == 2)[1].size > 0:
                j = np.where(y == 2)[1][0]  # Index of Y where y=2
                x = x[:,j]
                z = x ** 2
                assert np.all(z == Z[:,j])
                assert type(z) is GF
                assert z.dtype == dtype

    def test_log(self, log):
        GF, X, Z = log["GF"], log["X"], log["Z"]
        if GF.order > 2**16:  # TODO: Skip slow log() for very large fields
            return
        for dtype in GF.dtypes:
            x = X.astype(dtype)
            z = np.log(x)
            assert np.all(z == Z)


# class TestArithmeticNonField:

#     def test_multiple_add_int_scalar(self, multiple_add):
#         X, Y, Z = multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
#         i = np.random.randint(0, Z.shape[0], 10)  # Random x indices
#         j = np.random.randint(0, Z.shape[1])  # Random y index
#         x = X[i,0]
#         y = Y[0,j]
#         assert np.all(x * y == Z[i,j])
#         assert np.all(y * x == Z[i,j])

#     def test_multiple_add_int_array(self, multiple_add):
#         X, Y, Z = multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
#         i = np.random.randint(0, Z.shape[0], 10)  # Random x indices
#         j = np.random.randint(0, Z.shape[1])  # Random y index
#         x = X[i,0]
#         y = Y[0,j]
#         assert np.all(x * y == Z[i,j])
#         assert np.all(y * x == Z[i,j])

#     def test_rmul_int_scalar(self, multiple_add):
#         GF, X, Y, Z = multiple_add["GF"], multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
#         i = np.random.randint(0, Z.shape[0])  # Random x index
#         j = np.random.randint(0, Z.shape[1])  # Random y index

#         x = X.copy()
#         y = Y[i,j]  # Integer, non-field element
#         x[i,j] *= y
#         assert x[i,j] == Z[i,j]
#         assert isinstance(x, GF)

#         # TODO: Should this work?
#         # x = X
#         # y = Y[i,j].copy()  # Integer, non-field element
#         # y *= x[i,j]
#         # assert y == Z[i,j]
#         # assert isinstance(y, GF)

#     def test_rmul_int_array(self, multiple_add):
#         GF, X, Y, Z = multiple_add["GF"], multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
#         i = np.random.randint(0, Z.shape[0])  # Random x index

#         x = X.copy()
#         y = Y[i,:]
#         x[i,:] *= y
#         assert np.all(x[i,:] == Z[i,:])
#         assert isinstance(x, GF)

#         x = X
#         y = Y[i,:].copy()
#         y *= x[i,:]
#         assert np.all(y == Z[i,:])
#         assert isinstance(y, GF)


class TestExceptions:
    def test_add_int_scalar(self, field):
        x = field.Random(10)
        y = int(randint(0, field.order, 1, field.dtypes[-1]))
        with pytest.raises(TypeError):
            z = x + y
        with pytest.raises(TypeError):
            z = y + x

    def test_add_int_array(self, field):
        x = field.Random(10)
        y = randint(0, field.order, 10, field.dtypes[-1])
        with pytest.raises(TypeError):
            z = x + y
        with pytest.raises(TypeError):
            z = y + x

    def test_right_add_int_scalar(self, field):
        x = field.Random(10)
        y = int(randint(0, field.order, 1, field.dtypes[-1]))
        with pytest.raises(TypeError):
            x += y
        with pytest.raises(TypeError):
            y += x

    def test_right_add_int_array(self, field):
        x = field.Random(10)
        y = randint(0, field.order, 10, field.dtypes[-1])
        with pytest.raises(TypeError):
            x += y
        with pytest.raises(TypeError):
            y += x

    def test_subtract_int_scalar(self, field):
        x = field.Random(10)
        y = int(randint(0, field.order, 1, field.dtypes[-1]))
        with pytest.raises(TypeError):
            z = x - y
        with pytest.raises(TypeError):
            z = y - x

    def test_subtract_int_array(self, field):
        x = field.Random(10)
        y = randint(0, field.order, 10, field.dtypes[-1])
        with pytest.raises(TypeError):
            z = x - y
        with pytest.raises(TypeError):
            z = y - x

    def test_right_subtract_int_scalar(self, field):
        x = field.Random(10)
        y = int(randint(0, field.order, 1, field.dtypes[-1]))
        with pytest.raises(TypeError):
            x -= y
        with pytest.raises(TypeError):
            y -= x

    def test_right_subtract_int_array(self, field):
        x = field.Random(10)
        y = randint(0, field.order, 10, field.dtypes[-1])
        with pytest.raises(TypeError):
            x -= y
        with pytest.raises(TypeError):
            y -= x

    # NOTE: Don't test multiply with integer because that is a valid operation, namely "multiple addition"

    def test_divide_int_scalar(self, field):
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

    def test_divide_int_array(self, field):
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

    def test_right_divide_int_scalar(self, field):
        x = field.Random(10)
        y = int(randint(1, field.order, 1, field.dtypes[-1]))
        with pytest.raises(TypeError):
            x /= y
        with pytest.raises(TypeError):
            x //= y
        with pytest.raises(TypeError):
            y /= x
        with pytest.raises(TypeError):
            y //= x

    def test_right_divide_int_array(self, field):
        x = field.Random(10)
        y = randint(1, field.order, 10, field.dtypes[-1])
        with pytest.raises(TypeError):
            x /= y
        with pytest.raises(TypeError):
            x //= y
        with pytest.raises(TypeError):
            y /= x
        with pytest.raises(TypeError):
            y //= x

    def test_divide_by_zero(self, field):
        x = field.Random(10)
        with pytest.raises(ZeroDivisionError):
            y = field(0)
            z = x / y
        with pytest.raises(ZeroDivisionError):
            y = field.Random(10)
            y[0] = 0  # Ensure one value is zero
            z = x / y

    def test_multiplicative_inverse_of_zero(self, field):
        x = field.Random(10)
        x[0] = 0  # Ensure one value is zero
        with pytest.raises(ZeroDivisionError):
            z = x ** -1

    # NOTE: Don't test power to integer because that's valid

    def test_zero_to_negative_power(self, field):
        x = field.Random(10)
        x[0] = 0  # Ensure one value is zero
        with pytest.raises(ZeroDivisionError):
            y = -3
            z = x ** y
        with pytest.raises(ZeroDivisionError):
            y = -3*np.ones(x.size, field.dtypes[-1])
            z = x ** y

    def test_log_of_zero(self, field):
        with pytest.raises(ArithmeticError):
            x = field(0)
            z = np.log(x)
        with pytest.raises(ArithmeticError):
            x = field.Random(10)
            x[0] = 0  # Ensure one value is zero
            z = np.log(x)
