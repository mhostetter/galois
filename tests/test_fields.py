"""
A pytest module to test Galois field array classes.
"""
import pytest
import numpy as np

import galois


class TestInstantiation:
    def test_list_int(self, field):
        v = [0,1,0,1]
        a = field(v)
        check_array(a, field)

    def test_list_int_out_of_range(self, field):
        v = [0,1,0,field.order]
        with pytest.raises(AssertionError):
            a = field(v)
        v = [0,1,0,-1]
        with pytest.raises(AssertionError):
            a = field(v)

    def test_list_float(self, field):
        v = [0.1, field.order-0.8, 0.1, field.order-0.8]
        with pytest.raises(AssertionError):
            a = field(v)

    def test_array_int(self, field):
        v = np.array([0,1,0,1])
        a = field(v)
        check_array(a, field)

    def test_array_int_out_of_range(self, field):
        v = np.array([0,1,0,field.order])
        with pytest.raises(AssertionError):
            a = field(v)
        v = np.array([0,1,0,-1])
        with pytest.raises(AssertionError):
            a = field(v)

    def test_array_float(self, field):
        v = np.array([0.1, field.order-0.8, 0.1, field.order-0.8])
        with pytest.raises(AssertionError):
            a = field(v)

    def test_zeros(self, field):
        a = field.Zeros(10)
        assert np.all(a == 0)
        check_array(a, field)

    def test_ones(self, field):
        a = field.Ones(10)
        assert np.all(a == 1)
        check_array(a, field)

    def test_random(self, field):
        a = field.Random(10)
        assert np.all(a >= 0) and np.all(a < field.order)
        check_array(a, field)

    def test_random_element(self, field):
        a = field.Random()
        assert 0 <= a < field.order
        assert a.ndim == 0
        check_array(a, field)


class TestView:
    def test_array_correct_dtype(self, field):
        v = np.array([0,1,0,1], dtype=field._dtype)
        a = v.view(field)

    def test_array_incorrect_dtype(self, field):
        v = np.array([0,1,0,1], dtype=float)
        with pytest.raises(AssertionError):
            a = v.view(field)

    def test_array_out_of_range(self, field):
        v = np.array([0,1,0,field.order], dtype=field._dtype)
        with pytest.raises(AssertionError):
            a = v.view(field)


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
        with pytest.raises(AssertionError):
            a[0] = field.order

    def test_slice_constant_out_of_range(self, field):
        a = field([0,1,0,1])
        with pytest.raises(AssertionError):
            a[0:2] = field.order

    def test_slice_list_out_of_range(self, field):
        a = field([0,1,0,1])
        with pytest.raises(AssertionError):
            a[0:2] = [field.order, field.order]

    def test_slice_array_out_of_range(self, field):
        a = field([0,1,0,1])
        with pytest.raises(AssertionError):
            a[0:2] = np.array([field.order, field.order])


class TestArithmetic:
    def test_add(self, add):
        x = add["X"]
        y = add["Y"]
        z = x + y
        assert np.all(z == add["Z"])
        check_array(z, add["GF"])

    def test_subtract(self, subtract):
        x = subtract["X"]
        y = subtract["Y"]
        z = x - y
        assert np.all(z == subtract["Z"])
        check_array(z, subtract["GF"])

    def test_multiply(self, multiply):
        x = multiply["X"]
        y = multiply["Y"]
        z = x * y
        assert np.all(z == multiply["Z"])
        check_array(z, multiply["GF"])

    def test_divison(self, divison):
        x = divison["X"]
        y = divison["Y"]
        z = x / y
        assert np.all(z == divison["Z"])
        check_array(z, divison["GF"])

    def test_additive_inverse(self, additive_inverse):
        x = additive_inverse["X"]
        z = -x
        assert np.all(z == additive_inverse["Z"])
        check_array(z, additive_inverse["GF"])

    def test_multiplicative_inverse(self, multiplicative_inverse):
        x = multiplicative_inverse["X"]
        z = 1 / x
        assert np.all(z == multiplicative_inverse["Z"])
        check_array(z, multiplicative_inverse["GF"])

    def test_power(self, power):
        x = power["X"]
        y = power["Y"]
        z = x ** y
        assert np.all(z == power["Z"])
        check_array(z, power["GF"])

    def test_square(self, power):
        # Not guaranteed to have y=2 for "sparse" LUTs
        if np.where(power["Y"] == 2)[1].size > 0:
            j = np.where(power["Y"] == 2)[1][0]  # Index of Y where y=2
            x = power["X"][:,j]
            z = x ** 2
            assert np.all(z == power["Z"][:,j])
            check_array(z, power["GF"])

    def test_log(self, log):
        x = log["X"]
        z = np.log(x)
        assert np.all(z == log["Z"])


class TestArithmeticNonField:
    def test_add_int_scalar(self, add):
        shape = (10)
        i = np.random.randint(0, add["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, add["Z"].shape[1])  # Random y index
        x = add["X"][i,0]
        y = int(add["Y"][0,j])
        assert np.all(x + y == add["Z"][i,j])
        assert np.all(y + x == add["Z"][i,j])

    def test_add_int_array(self, add):
        shape = (10)
        i = np.random.randint(0, add["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, add["Z"].shape[1])  # Random y index
        x = add["X"][i,0]
        y = np.array(add["Y"][0,j], dtype=int)
        assert np.all(x + y == add["Z"][i,j])
        assert np.all(y + x == add["Z"][i,j])

    def test_subtract_int_scalar(self, subtract):
        shape = (10)

        i = np.random.randint(0, subtract["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, subtract["Z"].shape[1])  # Random y index
        x = subtract["X"][i,0]
        y = int(subtract["Y"][0,j])
        assert np.all(x - y == subtract["Z"][i,j])

        i = np.random.randint(0, subtract["Z"].shape[0])  # Random x indices
        j = np.random.randint(0, subtract["Z"].shape[1], shape)  # Random y index
        x = int(subtract["X"][i,0])
        y = subtract["Y"][0,j]
        assert np.all(x - y == subtract["Z"][i,j])

    def test_subtract_int_array(self, subtract):
        shape = (10)

        i = np.random.randint(0, subtract["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, subtract["Z"].shape[1], shape)  # Random y indices
        x = subtract["X"][i,0]
        y = np.array(subtract["Y"][0,j], dtype=int)
        assert np.all(x - y == subtract["Z"][i,j])

        i = np.random.randint(0, subtract["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, subtract["Z"].shape[1], shape)  # Random y indices
        x = np.array(subtract["X"][i,0], dtype=int)
        y = subtract["Y"][0,j]
        assert np.all(x - y == subtract["Z"][i,j])

    def test_multiply_int_scalar(self, multiply):
        shape = (10)
        i = np.random.randint(0, multiply["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, multiply["Z"].shape[1])  # Random y index
        x = multiply["X"][i,0]
        y = int(multiply["Y"][0,j])
        assert np.all(x * y == multiply["Z"][i,j])
        assert np.all(y * x == multiply["Z"][i,j])

    def test_multiply_int_array(self, multiply):
        shape = (10)
        i = np.random.randint(0, multiply["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, multiply["Z"].shape[1])  # Random y index
        x = multiply["X"][i,0]
        y = np.array(multiply["Y"][0,j], dtype=int)
        assert np.all(x * y == multiply["Z"][i,j])
        assert np.all(y * x == multiply["Z"][i,j])

    def test_divison_int_scalar(self, divison):
        shape = (10)

        i = np.random.randint(0, divison["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, divison["Z"].shape[1])  # Random y index
        x = divison["X"][i,0]
        y = int(divison["Y"][0,j])
        assert np.all(x / y == divison["Z"][i,j])

        i = np.random.randint(0, divison["Z"].shape[0])  # Random x index
        j = np.random.randint(0, divison["Z"].shape[1], shape)  # Random y indices
        x = int(divison["X"][i,0])
        y = divison["Y"][0,j]
        assert np.all(x / y == divison["Z"][i,j])

    def test_divison_int_array(self, divison):
        shape = (10)

        i = np.random.randint(0, divison["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, divison["Z"].shape[1], shape)  # Random y indices
        x = divison["X"][i,0]
        y = np.array(divison["Y"][0,j], dtype=int)
        assert np.all(x / y == divison["Z"][i,j])

        i = np.random.randint(0, divison["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, divison["Z"].shape[1], shape)  # Random y indices
        x = np.array(divison["X"][i,0], dtype=int)
        y = divison["Y"][0,j]
        assert np.all(x / y == divison["Z"][i,j])

    def test_radd_scalar(self, add):
        GF = add["GF"]
        i = np.random.randint(0, add["Z"].shape[0])  # Random x index
        j = np.random.randint(0, add["Z"].shape[1])  # Random y index
        x = GF(add["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = add["Y"][i,j]
        x[i,j] += y
        assert x[i,j] == add["Z"][i,j]

    def test_radd_vector(self, add):
        GF = add["GF"]
        i = np.random.randint(0, add["Z"].shape[0])  # Random x index
        x = GF(add["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = add["Y"][i,:]
        x[i,:] += y
        assert np.all(x[i,:] == add["Z"][i,:])

    def test_radd_matrix(self, add):
        GF = add["GF"]
        x = GF(add["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = add["Y"][:,:]
        x[:,:] += y
        assert np.all(x[:,:] == add["Z"][:,:])

    def test_rsub_scalar(self, subtract):
        GF = subtract["GF"]
        i = np.random.randint(0, subtract["Z"].shape[0])  # Random x index
        j = np.random.randint(0, subtract["Z"].shape[1])  # Random y index
        x = GF(subtract["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = subtract["Y"][i,j]
        x[i,j] -= y
        assert x[i,j] == subtract["Z"][i,j]

    def test_rsub_vector(self, subtract):
        GF = subtract["GF"]
        i = np.random.randint(0, subtract["Z"].shape[0])  # Random x index
        x = GF(subtract["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = subtract["Y"][i,:]
        x[i,:] -= y
        assert np.all(x[i,:] == subtract["Z"][i,:])

    def test_rsub_matrix(self, subtract):
        GF = subtract["GF"]
        x = GF(subtract["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = subtract["Y"][:,:]
        x[:,:] -= y
        assert np.all(x[:,:] == subtract["Z"][:,:])

    def test_rmul_scalar(self, multiply):
        GF = multiply["GF"]
        i = np.random.randint(0, multiply["Z"].shape[0])  # Random x index
        j = np.random.randint(0, multiply["Z"].shape[1])  # Random y index
        x = GF(multiply["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = multiply["Y"][i,j]
        x[i,j] *= y
        assert x[i,j] == multiply["Z"][i,j]

    def test_rmul_vector(self, multiply):
        GF = multiply["GF"]
        i = np.random.randint(0, multiply["Z"].shape[0])  # Random x index
        x = GF(multiply["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = multiply["Y"][i,:]
        x[i,:] *= y
        assert np.all(x[i,:] == multiply["Z"][i,:])

    def test_rmul_matrix(self, multiply):
        GF = multiply["GF"]
        x = GF(multiply["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = multiply["Y"][:,:]
        x[:,:] *= y
        assert np.all(x[:,:] == multiply["Z"][:,:])

    def test_rdiv_scalar(self, divison):
        GF = divison["GF"]
        i = np.random.randint(0, divison["Z"].shape[0])  # Random x index
        j = np.random.randint(0, divison["Z"].shape[1])  # Random y index
        x = GF(divison["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = divison["Y"][i,j]
        x[i,j] /= y
        assert x[i,j] == divison["Z"][i,j]

    def test_rdiv_vector(self, divison):
        GF = divison["GF"]
        i = np.random.randint(0, divison["Z"].shape[0])  # Random x index
        x = GF(divison["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = divison["Y"][i,:]
        x[i,:] /= y
        assert np.all(x[i,:] == divison["Z"][i,:])

    def test_rdiv_matrix(self, divison):
        GF = divison["GF"]
        x = GF(divison["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = divison["Y"][:,:]
        x[:,:] /= y
        assert np.all(x[:,:] == divison["Z"][:,:])


class TestArithmeticExceptions:
    def test_divide_by_zero(self, field):
        x = field.Random(field.order)
        with pytest.raises(ZeroDivisionError):
            y = 0
            z = x / y
        with pytest.raises(ZeroDivisionError):
            y = np.arange(0, field.order, dtype=int)
            z = x / y
        with pytest.raises(ZeroDivisionError):
            y = field.Elements()
            z = x / y

    def test_multiplicative_inverse_of_zero(self, field):
        x = field.Elements()
        with pytest.raises(ZeroDivisionError):
            z = x ** -1

    def test_zero_to_negative_power(self, field):
        x = field.Elements()
        with pytest.raises(ZeroDivisionError):
            y = -3
            z = x ** y
        with pytest.raises(ZeroDivisionError):
            y = -3*np.ones(x.size, dtype=int)
            z = x ** y

    def test_log_of_zero(self, field):
        x = field.Elements()
        with pytest.raises(ArithmeticError):
            z = np.log(x)
        with pytest.raises(ArithmeticError):
            z = np.log(field(0))


class TestArithmeticTypes:
    def test_scalar_int_return_type(self, field):
        shape = ()
        ndim = 0
        a = field.Random(shape)
        b = 1
        c = a + b
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape
        c = b + a
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape

    def test_vector_int_return_type(self, field):
        shape = (10,)
        ndim = 1
        a = field.Random(shape)
        b = 1
        c = a + b
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape
        c = b + a
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape

    def test_matrix_int_return_type(self, field):
        shape = (10,10)
        ndim = 2
        a = field.Random(shape)
        b = 1
        c = a + b
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape
        c = b + a
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape

    def test_scalar_scalar_return_type(self, field):
        shape = ()
        ndim = 0
        a = field.Random(shape)
        b = field.Random(shape)
        c = a + b
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape

    def test_vector_vector_return_type(self, field):
        shape = (10,)
        ndim = 1
        a = field.Random(shape)
        b = field.Random(shape)
        c = a + b
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape

    def test_matrix_matrix_return_type(self, field):
        shape = (10,10)
        ndim = 2
        a = field.Random(shape)
        b = field.Random(shape)
        c = a + b
        assert type(c) is field
        assert c.ndim == ndim
        assert c.shape == shape

    def test_scalar_int_out_of_range(self, field):
        shape = ()
        a = field.Random(shape)
        b = field.order
        with pytest.raises(ValueError):
            c = a + b
        with pytest.raises(ValueError):
            c = b + a

    def test_vector_int_out_of_range(self, field):
        shape = (10,)
        a = field.Random(shape)
        b = field.order
        with pytest.raises(ValueError):
            c = a + b
        with pytest.raises(ValueError):
            c = b + a

    def test_matrix_int_out_of_range(self, field):
        shape = (10,10)
        a = field.Random(shape)
        b = field.order
        with pytest.raises(ValueError):
            c = a + b
        with pytest.raises(ValueError):
            c = b + a

    def test_scalar_scalar_out_of_range(self, field):
        shape = ()
        a = field.Random(shape)
        b = field.order*np.ones(shape, dtype=int)
        with pytest.raises(ValueError):
            c = a + b
        # TODO: Can't figure out how to make this fail
        # with pytest.raises(ValueError):
        #     c = b + a

    def test_vector_vector_out_of_range(self, field):
        shape = (10,)
        a = field.Random(shape)
        b = field.order*np.ones(shape, dtype=int)
        with pytest.raises(ValueError):
            c = a + b
        with pytest.raises(ValueError):
            c = b + a

    def test_matrix_matrix_out_of_range(self, field):
        shape = (10,10)
        a = field.Random(shape)
        b = field.order*np.ones(shape, dtype=int)
        with pytest.raises(ValueError):
            c = a + b
        with pytest.raises(ValueError):
            c = b + a


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

    def test_reduce_divison(self, field):
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
        b = np.random.randint(0, field.order, 1)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.multiply.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] * b
        assert np.all(a == a_truth)

    def test_at_divide(self, field):
        a = field.Random(10)
        b = np.random.randint(1, field.order, 1)
        idxs = [0,1,1,4,8]
        a_truth = field(a)  # Ensure a copy happens
        np.true_divide.at(a, idxs, b)
        for i in idxs:
            a_truth[i] = a_truth[i] / b
        assert np.all(a == a_truth)

        a = field.Random(10)
        b = np.random.randint(1, field.order, 1)
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
    def test_characteristic(self, field):
        a = field.Elements()
        p = field.characteristic
        b = a * p
        assert np.all(b == 0)

    def test_property_2(self, field):
        a = field.Elements()[1:]
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
        p = field.characteristic
        for a in field.Elements():
            poly = poly * galois.Poly([1, -a], field=field)
        assert poly == galois.Poly.NonZero([1, -1], [p, 1], field=field)


def check_array(array, GF):
    assert type(array) is GF
    assert array.dtype == GF._dtype
