"""
A pytest module to test Galois field array classes.
"""
import pytest
import random
import numpy as np

import galois


class TestFieldClasses:
    def test_ufunc_attributes(self, field_classes):
        GF, mode = field_classes["GF"], field_classes["mode"]
        if mode == "auto":
            if GF.order == 2:
                mode = "calculate"
            elif GF.order <= 2**16:
                mode = "lookup"
            else:
                mode = "calcualte"

        assert GF.ufunc_target == "cpu"
        assert GF.ufunc_mode == mode


class TestInstantiation:
    def test_cant_instantiate_GFp(self):
        with pytest.raises(NotImplementedError):
            a = galois.GFp([1,2,3])

    def test_cant_instantiate_GF2m(self):
        with pytest.raises(NotImplementedError):
            a = galois.GF2m([1,2,3])

    def test_list_int(self, field):
        v = [0,1,0,1]
        a = field(v)
        assert type(a) is field

    def test_list_int_with_dtype(self, field):
        v = [0,1,0,1]
        dtype = random.choice(field.dtypes)
        a = field(v, dtype=dtype)
        assert a.dtype == dtype

    def test_list_int_out_of_range(self, field):
        v = [0,1,0,field.order]
        with pytest.raises(ValueError):
            a = field(v)
        v = [0,1,0,-1]
        with pytest.raises(ValueError):
            a = field(v)

    def test_list_float(self, field):
        v = [0.1, field.order-0.8, 0.1, field.order-0.8]
        with pytest.raises(TypeError):
            a = field(v)

    def test_array_int(self, field):
        v = np.array([0,1,0,1])
        a = field(v)
        assert type(a) is field

    def test_array_int_out_of_range(self, field):
        v = np.array([0,1,0,field.order])
        with pytest.raises(ValueError):
            a = field(v)
        v = np.array([0,1,0,-1])
        with pytest.raises(ValueError):
            a = field(v)

    def test_array_float(self, field):
        v = np.array([0.1, field.order-0.8, 0.1, field.order-0.8], dtype=float)
        with pytest.raises(TypeError):
            a = field(v)

    def test_zeros(self, field):
        a = field.Zeros(10)
        assert np.all(a == 0)
        assert type(a) is field

    def test_zeros_with_dtype(self, field):
        dtype = random.choice(field.dtypes)
        a = field.Zeros(10, dtype=dtype)
        assert np.all(a == 0)
        assert a.dtype == dtype

    def test_ones(self, field):
        a = field.Ones(10)
        assert np.all(a == 1)
        assert type(a) is field

    def test_ones_with_dtype(self, field):
        dtype = random.choice(field.dtypes)
        a = field.Ones(10, dtype=dtype)
        assert np.all(a == 1)
        assert a.dtype == dtype

    def test_random(self, field):
        a = field.Random(10)
        assert np.all(a >= 0) and np.all(a < field.order)
        assert type(a) is field

    def test_random_with_dtype(self, field):
        dtype = random.choice(field.dtypes)
        a = field.Random(10, dtype=dtype)
        assert np.all(a >= 0) and np.all(a < field.order)
        assert a.dtype == dtype

    def test_random_element(self, field):
        a = field.Random()
        assert 0 <= a < field.order
        assert a.ndim == 0
        assert type(a) is field

    def test_random_element_with_dtype(self, field):
        dtype = random.choice(field.dtypes)
        a = field.Random(dtype=dtype)
        assert 0 <= a < field.order
        assert a.ndim == 0
        assert type(a) is field
        assert a.dtype == dtype


class TestView:
    def test_array_valid_dtypes(self, field):
        for dtype in field.dtypes:
            v = np.array([0,1,0,1], dtype=dtype)
            a = v.view(field)

    def test_array_too_small_integer_dtype(self):
        GF = galois.GF_factory(3191, 1)
        v = np.array([0,1,0,1], dtype=np.int8)
        with pytest.raises(TypeError):
            a = v.view(GF)

    def test_array_non_valid_dtype(self, field):
        v = np.array([0,1,0,1], dtype=float)
        with pytest.raises(TypeError):
            a = v.view(field)

    def test_array_out_of_range_values(self, field):
        dtype = random.choice(field.dtypes)  # Random dtype that's compatible with this field
        v = np.array([0,1,0,field.order], dtype=dtype)
        with pytest.raises(ValueError):
            a = v.view(field)

    # def test_1(self, field, dtype):
    #     a = np.random.randint(0, field.order, 10, dtype=np.int16)
    #     ga = a.view(field)
    #     assert np.all(a == ga)
    #     assert ga.dtype is a.dtype


class TestAssignment:
    def test_index_constant(self, field):
        a = field([0,1,0,1])
        a[0] = 1

    def test_slice_constant(self, field):
        a = field([0,1,0,1])
        a[0:2] = 1

    def test_slice_list(self, field):
        a = field([0,1,0,1])
        a[0:2] = [1,1]

    def test_slice_array(self, field):
        a = field([0,1,0,1])
        a[0:2] = np.array([1,1])

    def test_index_constant_out_of_range(self, field):
        a = field([0,1,0,1])
        with pytest.raises(ValueError):
            a[0] = field.order

    def test_slice_constant_out_of_range(self, field):
        a = field([0,1,0,1])
        with pytest.raises(ValueError):
            a[0:2] = field.order

    def test_slice_list_out_of_range(self, field):
        a = field([0,1,0,1])
        with pytest.raises(ValueError):
            a[0:2] = [field.order, field.order]

    def test_slice_array_out_of_range(self, field):
        a = field([0,1,0,1])
        with pytest.raises(ValueError):
            a[0:2] = np.array([field.order, field.order])


class TestArithmetic:
    def test_add(self, add, dtype):
        GF, X, Y, Z = add["GF"], add["X"], add["Y"], add["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y.astype(dtype)
            z = x + y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_subtract(self, subtract, dtype):
        GF, X, Y, Z = subtract["GF"], subtract["X"], subtract["Y"], subtract["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y.astype(dtype)
            z = x - y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_multiply(self, multiply, dtype):
        GF, X, Y, Z = multiply["GF"], multiply["X"], multiply["Y"], multiply["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y.astype(dtype)
            z = x * y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_divide(self, divide, dtype):
        GF, X, Y, Z = divide["GF"], divide["X"], divide["Y"], divide["Z"]
        if dtype in GF.dtypes:
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

    def test_additive_inverse(self, additive_inverse, dtype):
        GF, X, Z = additive_inverse["GF"], additive_inverse["X"], additive_inverse["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)
            z = -x
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_multiplicative_inverse(self, multiplicative_inverse, dtype):
        GF, X, Z = multiplicative_inverse["GF"], multiplicative_inverse["X"], multiplicative_inverse["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)

            z = GF(1) / x
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == np.int64  # TODO: Re-investigate this

            z = GF(1) // x
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == np.int64  # TODO: Re-investigate this

            z = x ** -1
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_multiple_add(self, multiple_add, dtype):
        GF, X, Y, Z = multiple_add["GF"], multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y  # Don't convert this, it's not a field element
            z = x * y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_power(self, power, dtype):
        GF, X, Y, Z = power["GF"], power["X"], power["Y"], power["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)
            y = Y  # Don't convert this, it's not a field element
            z = x ** y
            assert np.all(z == Z)
            assert type(z) is GF
            assert z.dtype == dtype

    def test_power_zero_to_zero(self, field, dtype):
        if dtype in field.dtypes:
            x = field.Zeros(10, dtype=dtype)
            y = np.zeros(10, dtype=int)
            z = x ** y
            Z = np.ones(10, dtype=int)
            assert np.all(z == Z)
            assert type(z) is field
            assert z.dtype == dtype

    def test_power_zero_to_positive_integer(self, field, dtype):
        if dtype in field.dtypes:
            x = field.Zeros(10, dtype=dtype)
            y = np.random.randint(1, 2*field.order, 10, dtype=int)
            z = x ** y
            Z = np.zeros(10, dtype=int)
            assert np.all(z == Z)
            assert type(z) is field
            assert z.dtype == dtype

    def test_square(self, power, dtype):
        GF, X, Y, Z = power["GF"], power["X"], power["Y"], power["Z"]
        if dtype in GF.dtypes:
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

    def test_log(self, log, dtype):
        GF, X, Z = log["GF"], log["X"], log["Z"]
        if dtype in GF.dtypes:
            x = X.astype(dtype)
            z = np.log(x)
            assert np.all(z == Z)


class TestArithmeticNonField:
    shape = (10,)

    def test_add_int_scalar(self, field):
        x = field.Random(self.shape)
        y = int(np.random.randint(0, field.order, 1, dtype=int))
        with pytest.raises(TypeError):
            z = x + y
        with pytest.raises(TypeError):
            z = y + x

    def test_add_int_array(self, field):
        x = field.Random(self.shape)
        y = np.random.randint(0, field.order, self.shape, dtype=int)
        with pytest.raises(TypeError):
            z = x + y
        with pytest.raises(TypeError):
            z = y + x

    def test_subtract_int_scalar(self, field):
        x = field.Random(self.shape)
        y = int(np.random.randint(0, field.order, 1, dtype=int))
        with pytest.raises(TypeError):
            z = x - y
        with pytest.raises(TypeError):
            z = y - x

    def test_subtract_int_array(self, field):
        x = field.Random(self.shape)
        y = np.random.randint(0, field.order, self.shape, dtype=int)
        with pytest.raises(TypeError):
            z = x - y
        with pytest.raises(TypeError):
            z = y - x

    def test_multiple_add_int_scalar(self, multiple_add):
        X, Y, Z = multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
        i = np.random.randint(0, Z.shape[0], self.shape)  # Random x indices
        j = np.random.randint(0, Z.shape[1])  # Random y index
        x = X[i,0]
        y = Y[0,j]
        assert np.all(x * y == Z[i,j])
        assert np.all(y * x == Z[i,j])

    def test_multiple_add_int_array(self, multiple_add):
        X, Y, Z = multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
        i = np.random.randint(0, Z.shape[0], self.shape)  # Random x indices
        j = np.random.randint(0, Z.shape[1])  # Random y index
        x = X[i,0]
        y = Y[0,j]
        assert np.all(x * y == Z[i,j])
        assert np.all(y * x == Z[i,j])

    def test_divide_int_scalar(self, field):
        x = field.Random(self.shape, low=1)
        y = int(np.random.randint(1, field.order, 1, dtype=int))
        with pytest.raises(TypeError):
            z = x / y
        with pytest.raises(TypeError):
            z = x // y
        with pytest.raises(TypeError):
            z = y / x
        with pytest.raises(TypeError):
            z = y // x

    def test_divide_int_array(self, field):
        x = field.Random(self.shape, low=1)
        y = np.random.randint(1, field.order, self.shape, dtype=int)
        with pytest.raises(TypeError):
            z = x / y
        with pytest.raises(TypeError):
            z = x // y
        with pytest.raises(TypeError):
            z = y / x
        with pytest.raises(TypeError):
            z = y // x

    def test_radd_int_scalar(self, field):
        x = field.Random(self.shape)
        y = int(np.random.randint(0, field.order, 1, dtype=int))
        with pytest.raises(TypeError):
            x += y
        with pytest.raises(TypeError):
            y += x

    def test_radd_int_array(self, field):
        x = field.Random(self.shape)
        y = np.random.randint(0, field.order, self.shape, dtype=int)
        with pytest.raises(TypeError):
            x += y
        with pytest.raises(TypeError):
            y += x

    def test_rsub_int_scalar(self, field):
        x = field.Random(self.shape)
        y = int(np.random.randint(0, field.order, 1, dtype=int))
        with pytest.raises(TypeError):
            x -= y
        with pytest.raises(TypeError):
            y -= x

    def test_rsub_int_array(self, field):
        x = field.Random(self.shape)
        y = np.random.randint(0, field.order, self.shape, dtype=int)
        with pytest.raises(TypeError):
            x -= y
        with pytest.raises(TypeError):
            y -= x

    def test_rmul_int_scalar(self, multiple_add):
        GF, X, Y, Z = multiple_add["GF"], multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
        i = np.random.randint(0, Z.shape[0])  # Random x index
        j = np.random.randint(0, Z.shape[1])  # Random y index

        x = X.copy()
        y = Y[i,j]  # Integer, non-field element
        x[i,j] *= y
        assert x[i,j] == Z[i,j]
        assert isinstance(x, GF)

        # TODO: Should this work?
        # x = X
        # y = Y[i,j].copy()  # Integer, non-field element
        # y *= x[i,j]
        # assert y == Z[i,j]
        # assert isinstance(y, GF)

    def test_rmul_int_array(self, multiple_add):
        GF, X, Y, Z = multiple_add["GF"], multiple_add["X"], multiple_add["Y"], multiple_add["Z"]
        i = np.random.randint(0, Z.shape[0])  # Random x index

        x = X.copy()
        y = Y[i,:]
        x[i,:] *= y
        assert np.all(x[i,:] == Z[i,:])
        assert isinstance(x, GF)

        x = X
        y = Y[i,:].copy()
        y *= x[i,:]
        assert np.all(y == Z[i,:])
        assert isinstance(y, GF)

    def test_rdiv_int_scalar(self, field):
        x = field.Random(self.shape)
        y = int(np.random.randint(1, field.order, 1, dtype=int))
        with pytest.raises(TypeError):
            x /= y
        with pytest.raises(TypeError):
            x //= y
        with pytest.raises(TypeError):
            y /= x
        with pytest.raises(TypeError):
            y //= x

    def test_rdiv_int_array(self, field):
        x = field.Random(self.shape)
        y = np.random.randint(1, field.order, self.shape, dtype=int)
        with pytest.raises(TypeError):
            x /= y
        with pytest.raises(TypeError):
            x //= y
        with pytest.raises(TypeError):
            y /= x
        with pytest.raises(TypeError):
            y //= x


class TestArithmeticExceptions:
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
            z = field(1) / x
        with pytest.raises(ZeroDivisionError):
            z = x ** -1

    def test_zero_to_negative_power(self, field):
        x = field.Random(10)
        x[0] = 0  # Ensure one value is zero
        with pytest.raises(ZeroDivisionError):
            y = -3
            z = x ** y
        with pytest.raises(ZeroDivisionError):
            y = -3*np.ones(x.size, dtype=int)
            z = x ** y

    def test_log_of_zero(self, field):
        with pytest.raises(ArithmeticError):
            x = field(0)
            z = np.log(x)
        with pytest.raises(ArithmeticError):
            x = field.Random(10)
            x[0] = 0  # Ensure one value is zero
            z = np.log(x)


# class TestArithmeticTypes:
#     def test_scalar_int_return_type(self, field):
#         shape = ()
#         ndim = 0
#         a = field.Random(shape)
#         b = 1
#         c = a + b
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape
#         c = b + a
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape

#     def test_vector_int_return_type(self, field):
#         shape = (10,)
#         ndim = 1
#         a = field.Random(shape)
#         b = 1
#         c = a + b
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape
#         c = b + a
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape

#     def test_matrix_int_return_type(self, field):
#         shape = (10,10)
#         ndim = 2
#         a = field.Random(shape)
#         b = 1
#         c = a + b
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape
#         c = b + a
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape

#     def test_scalar_scalar_return_type(self, field):
#         shape = ()
#         ndim = 0
#         a = field.Random(shape)
#         b = field.Random(shape)
#         c = a + b
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape

#     def test_vector_vector_return_type(self, field):
#         shape = (10,)
#         ndim = 1
#         a = field.Random(shape)
#         b = field.Random(shape)
#         c = a + b
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape

#     def test_matrix_matrix_return_type(self, field):
#         shape = (10,10)
#         ndim = 2
#         a = field.Random(shape)
#         b = field.Random(shape)
#         c = a + b
#         assert type(c) is field
#         assert c.ndim == ndim
#         assert c.shape == shape

#     def test_scalar_int_out_of_range(self, field):
#         shape = ()
#         a = field.Random(shape)
#         b = field.order
#         with pytest.raises(ValueError):
#             c = a + b
#         with pytest.raises(ValueError):
#             c = b + a

#     def test_vector_int_out_of_range(self, field):
#         shape = (10,)
#         a = field.Random(shape)
#         b = field.order
#         with pytest.raises(ValueError):
#             c = a + b
#         with pytest.raises(ValueError):
#             c = b + a

#     def test_matrix_int_out_of_range(self, field):
#         shape = (10,10)
#         a = field.Random(shape)
#         b = field.order
#         with pytest.raises(ValueError):
#             c = a + b
#         with pytest.raises(ValueError):
#             c = b + a

#     def test_scalar_scalar_out_of_range(self, field):
#         shape = ()
#         a = field.Random(shape)
#         b = field.order*np.ones(shape, dtype=int)
#         with pytest.raises(ValueError):
#             c = a + b
#         # TODO: Can't figure out how to make this fail
#         # with pytest.raises(ValueError):
#         #     c = b + a

#     def test_vector_vector_out_of_range(self, field):
#         shape = (10,)
#         a = field.Random(shape)
#         b = field.order*np.ones(shape, dtype=int)
#         with pytest.raises(ValueError):
#             c = a + b
#         with pytest.raises(ValueError):
#             c = b + a

#     def test_matrix_matrix_out_of_range(self, field):
#         shape = (10,10)
#         a = field.Random(shape)
#         b = field.order*np.ones(shape, dtype=int)
#         with pytest.raises(ValueError):
#             c = a + b
#         with pytest.raises(ValueError):
#             c = b + a


class TestBroadcasting:

    # NOTE: We don't need to verify the arithmetic is correct here, that was done in TestArithmetic

    def test_matrix_and_constant(self, field):
        A = field.Random((4,4))
        b = field.Random()
        B = b * field.Ones((4,4))
        assert np.all(A + b == A + B)


class TestUfuncMethods:
    def test_reduce_add(self, field):
        a = field.Random(10)
        b = np.add.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth + ai
        assert b == b_truth

    def test_reduce_subtract(self, field):
        a = field.Random(10)
        b = np.subtract.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth - ai
        assert b == b_truth

    def test_reduce_multiply(self, field):
        a = field.Random(10)
        b = np.multiply.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth * ai
        assert b == b_truth

    def test_reduce_divide(self, field):
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

    def test_reduce_power(self, field):
        a = field.Random(10)
        b = np.power.reduce(a)
        b_truth = a[0]
        for ai in a[1:]:
            b_truth = b_truth ** ai
        assert b == b_truth

    def test_accumulate_add(self, field):
        a = field.Random(10)
        b = np.add.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] + a[i]
        assert np.all(b == b_truth)

    def test_accumulate_subtract(self, field):
        a = field.Random(10)
        b = np.subtract.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] - a[i]
        assert np.all(b == b_truth)

    def test_accumulate_multiply(self, field):
        a = field.Random(10)
        b = np.multiply.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] * a[i]
        assert np.all(b == b_truth)

    def test_accumulate_divide(self, field):
        a = field.Random(10, low=1)
        b = np.true_divide.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] / a[i]
        assert np.all(b == b_truth)

        a = field.Random(10, low=1)
        b = np.floor_divide.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] // a[i]
        assert np.all(b == b_truth)

    def test_accumulate_power(self, field):
        a = field.Random(10)
        b = np.power.accumulate(a)
        b_truth = field.Zeros(10)
        b_truth[0] = a[0]
        for i in range(1, 10):
            b_truth[i] = b_truth[i-1] ** a[i]
        assert np.all(b == b_truth)

    def test_reduceat_add(self, field):
        a = field.Random(10)
        idxs = [1,4,5,8]
        b = np.add.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.add.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.add.reduce(a[idxs[i]:idxs[i+1]])
        assert np.all(b == b_truth)

    def test_reduceat_subtract(self, field):
        a = field.Random(10)
        idxs = [1,4,5,8]
        b = np.subtract.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.subtract.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.subtract.reduce(a[idxs[i]:idxs[i+1]])
        assert np.all(b == b_truth)

    def test_reduceat_multiply(self, field):
        a = field.Random(10)
        idxs = [1,4,5,8]
        b = np.multiply.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.multiply.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.multiply.reduce(a[idxs[i]:idxs[i+1]])
        assert np.all(b == b_truth)

    def test_reduceat_divide(self, field):
        a = field.Random(10, low=1)
        idxs = [1,4,5,8]
        b = np.true_divide.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.true_divide.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.true_divide.reduce(a[idxs[i]:idxs[i+1]])
        assert np.all(b == b_truth)

        a = field.Random(10, low=1)
        idxs = [1,4,5,8]
        b = np.floor_divide.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.floor_divide.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.floor_divide.reduce(a[idxs[i]:idxs[i+1]])
        assert np.all(b == b_truth)

    def test_reduceat_power(self, field):
        a = field.Random(10)
        idxs = [1,4,5,8]
        b = np.power.reduceat(a, idxs)
        b_truth = field.Zeros(len(idxs))
        for i in range(len(idxs)):
            if i == len(idxs) - 1:
                b_truth[i] = np.power.reduce(a[idxs[i]:])
            else:
                b_truth[i] = np.power.reduce(a[idxs[i]:idxs[i+1]])
        assert np.all(b == b_truth)

    def test_outer_add(self, field):
        a = field.Random(10)
        b = field.Random(12)
        c = np.add.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] + b[j]
        assert np.all(c == c_truth)

    def test_outer_subtract(self, field):
        a = field.Random(10)
        b = field.Random(12)
        c = np.subtract.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] - b[j]
        assert np.all(c == c_truth)

    def test_outer_multiply(self, field):
        a = field.Random(10)
        b = np.random.randint(0, field.order, 12, dtype=int)
        c = np.multiply.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] * b[j]
        assert np.all(c == c_truth)

    def test_outer_divide(self, field):
        a = field.Random(10)
        b = field.Random(12, low=1)
        c = np.true_divide.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] / b[j]
        assert np.all(c == c_truth)

        a = field.Random(10)
        b = field.Random(12, low=1)
        c = np.floor_divide.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] // b[j]
        assert np.all(c == c_truth)

    def test_outer_power(self, field):
        a = field.Random(10)
        b = np.random.randint(1, field.order, 12)
        c = np.power.outer(a, b)
        c_truth = field.Zeros((a.size, b.size))
        for i in range(a.size):
            for j in range(b.size):
                c_truth[i,j] = a[i] ** b[j]
        assert np.all(c == c_truth)

    def test_at_add(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.add.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] + b
        assert np.all(a == a_truth)

    def test_at_subtract(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.subtract.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] - b
        assert np.all(a == a_truth)

    def test_at_multiply(self, field):
        a = field.Random(10)
        b = field.Random()
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.multiply.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] * b
        assert np.all(a == a_truth)

    def test_at_divide(self, field):
        a = field.Random(10)
        b = field.Random(low=1)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.true_divide.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] / b
        assert np.all(a == a_truth)

        a = field.Random(10)
        b = field.Random(low=1)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.floor_divide.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] // b
        assert np.all(a == a_truth)

    def test_at_negative(self, field):
        a = field.Random(10)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.negative.at(a, idxs)
        for i in idxs:
            a_truth[i] = -a_truth[i]
        assert np.all(a == a_truth)

    def test_at_power(self, field):
        a = field.Random(10)
        b = np.random.randint(1, field.order, 1)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.power.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] ** b
        assert np.all(a == a_truth)

    def test_at_log(self, field):
        a = field.Random(10, low=1)
        idxs = [0,1,4,8]  # Dont test index=1 twice like other tests because in GF(2) log(1)=0 and then log(0)=error
        a_truth = field(a)  # Ensure a copy happens
        np.log.at(a, idxs)
        for i in idxs:
            a_truth[i] = np.log(a_truth[i])
        assert np.all(a == a_truth)


class TestProperties:
    def test_properties(self, properties):
        GF = properties["GF"]
        assert GF.characteristic == properties["characteristic"]
        assert GF.degree == properties["degree"]
        assert GF.order == properties["order"]
        assert GF.alpha == properties["alpha"]
        assert all(GF.prim_poly.coeffs == properties["prim_poly"])

    def test_characteristic(self, field):
        if field.order < 2**16:
            a = field.Elements()
        else:
            # Only select some, not all, elements for very large fields
            a = field.Random(2**16)
        p = field.characteristic
        b = a * p
        assert np.all(b == 0)

    def test_property_2(self, field):
        if field.order < 2**16:
            a = field.Elements()[1:]
        else:
            # Only select some, not all, elements for very large fields
            a = field.Random(2**16, low=1)
        q = field.order
        assert np.all(a**q == a)

    def test_prim_poly(self, field):
        prim_poly = field.prim_poly  # Polynomial in GF(p)
        alpha = field.alpha
        poly = galois.Poly(prim_poly.coeffs, field=field)  # Polynomial in GF(p^m)
        assert poly(alpha) == 0

    def test_freshmans_dream(self, field):
        a = field.Random(10)
        b = field.Random(10)
        p = field.characteristic
        assert np.all((a + b)**p == a**p + b**p)

    def test_fermats_little_theorem(self, field):
        if field.order > 50:
            # Skip for very large fields because this takes too long
            return
        poly = galois.Poly([1], field=field)  # Base polynomial
        # p = field.characteristic
        for a in field.Elements():
            poly = poly * galois.Poly([1, -a], field=field)
        assert poly == galois.Poly.NonZero([1, -1], [field.order, 1], field=field)

    def test_exp_log_duality(self, field):
        alpha = field.alpha
        x = field.Random(10, low=1)
        e = np.log(x)
        assert np.all(alpha**e == x)
