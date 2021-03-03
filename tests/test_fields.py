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
        print(add)
        x = add["X"]
        y = add["Y"]
        z = x + y
        assert np.all(z == add["Z"])
        check_array(z, add["GF"])

    def test_sub(self, sub):
        x = sub["X"]
        y = sub["Y"]
        z = x - y
        assert np.all(z == sub["Z"])
        check_array(z, sub["GF"])

    def test_mul(self, mul):
        x = mul["X"]
        y = mul["Y"]
        z = x * y
        assert np.all(z == mul["Z"])
        check_array(z, mul["GF"])

    def test_div(self, div):
        x = div["X"]
        y = div["Y"]
        z = x / y
        assert np.all(z == div["Z"])
        check_array(z, div["GF"])

    def test_add_inv(self, add_inv):
        x = add_inv["X"]
        z = -x
        assert np.all(z == add_inv["Z"])
        check_array(z, add_inv["GF"])

    def test_mul_inv(self, mul_inv):
        x = mul_inv["X"]
        z = 1 / x
        assert np.all(z == mul_inv["Z"])
        check_array(z, mul_inv["GF"])

    def test_exp(self, exp):
        x = exp["X"]
        y = exp["Y"]
        z = x ** y
        assert np.all(z == exp["Z"])
        check_array(z, exp["GF"])

    def test_sqr(self, exp):
        j = np.where(exp["Y"] == 2)[1][0]  # Index of Y where y=2
        x = exp["X"][:,j]
        z = x ** 2
        assert np.all(z == exp["Z"][:,j])
        check_array(z, exp["GF"])

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

    def test_sub_int_scalar(self, sub):
        shape = (10)

        i = np.random.randint(0, sub["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, sub["Z"].shape[1])  # Random y index
        x = sub["X"][i,0]
        y = int(sub["Y"][0,j])
        assert np.all(x - y == sub["Z"][i,j])

        i = np.random.randint(0, sub["Z"].shape[0])  # Random x indices
        j = np.random.randint(0, sub["Z"].shape[1], shape)  # Random y index
        x = int(sub["X"][i,0])
        y = sub["Y"][0,j]
        assert np.all(x - y == sub["Z"][i,j])

    def test_sub_int_array(self, sub):
        shape = (10)

        i = np.random.randint(0, sub["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, sub["Z"].shape[1], shape)  # Random y indices
        x = sub["X"][i,0]
        y = np.array(sub["Y"][0,j], dtype=int)
        assert np.all(x - y == sub["Z"][i,j])

        i = np.random.randint(0, sub["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, sub["Z"].shape[1], shape)  # Random y indices
        x = np.array(sub["X"][i,0], dtype=int)
        y = sub["Y"][0,j]
        assert np.all(x - y == sub["Z"][i,j])

    def test_mul_int_scalar(self, mul):
        shape = (10)
        i = np.random.randint(0, mul["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, mul["Z"].shape[1])  # Random y index
        x = mul["X"][i,0]
        y = int(mul["Y"][0,j])
        assert np.all(x * y == mul["Z"][i,j])
        assert np.all(y * x == mul["Z"][i,j])

    def test_mul_int_array(self, mul):
        shape = (10)
        i = np.random.randint(0, mul["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, mul["Z"].shape[1])  # Random y index
        x = mul["X"][i,0]
        y = np.array(mul["Y"][0,j], dtype=int)
        assert np.all(x * y == mul["Z"][i,j])
        assert np.all(y * x == mul["Z"][i,j])

    def test_div_int_scalar(self, div):
        shape = (10)

        i = np.random.randint(0, div["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, div["Z"].shape[1])  # Random y index
        x = div["X"][i,0]
        y = int(div["Y"][0,j])
        assert np.all(x / y == div["Z"][i,j])

        i = np.random.randint(0, div["Z"].shape[0])  # Random x index
        j = np.random.randint(0, div["Z"].shape[1], shape)  # Random y indices
        x = int(div["X"][i,0])
        y = div["Y"][0,j]
        assert np.all(x / y == div["Z"][i,j])

    def test_div_int_array(self, div):
        shape = (10)

        i = np.random.randint(0, div["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, div["Z"].shape[1], shape)  # Random y indices
        x = div["X"][i,0]
        y = np.array(div["Y"][0,j], dtype=int)
        assert np.all(x / y == div["Z"][i,j])

        i = np.random.randint(0, div["Z"].shape[0], shape)  # Random x indices
        j = np.random.randint(0, div["Z"].shape[1], shape)  # Random y indices
        x = np.array(div["X"][i,0], dtype=int)
        y = div["Y"][0,j]
        assert np.all(x / y == div["Z"][i,j])

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

    def test_rsub_scalar(self, sub):
        GF = sub["GF"]
        i = np.random.randint(0, sub["Z"].shape[0])  # Random x index
        j = np.random.randint(0, sub["Z"].shape[1])  # Random y index
        x = GF(sub["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = sub["Y"][i,j]
        x[i,j] -= y
        assert x[i,j] == sub["Z"][i,j]

    def test_rsub_vector(self, sub):
        GF = sub["GF"]
        i = np.random.randint(0, sub["Z"].shape[0])  # Random x index
        x = GF(sub["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = sub["Y"][i,:]
        x[i,:] -= y
        assert np.all(x[i,:] == sub["Z"][i,:])

    def test_rsub_matrix(self, sub):
        GF = sub["GF"]
        x = GF(sub["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = sub["Y"][:,:]
        x[:,:] -= y
        assert np.all(x[:,:] == sub["Z"][:,:])

    def test_rmul_scalar(self, mul):
        GF = mul["GF"]
        i = np.random.randint(0, mul["Z"].shape[0])  # Random x index
        j = np.random.randint(0, mul["Z"].shape[1])  # Random y index
        x = GF(mul["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = mul["Y"][i,j]
        x[i,j] *= y
        assert x[i,j] == mul["Z"][i,j]

    def test_rmul_vector(self, mul):
        GF = mul["GF"]
        i = np.random.randint(0, mul["Z"].shape[0])  # Random x index
        x = GF(mul["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = mul["Y"][i,:]
        x[i,:] *= y
        assert np.all(x[i,:] == mul["Z"][i,:])

    def test_rmul_matrix(self, mul):
        GF = mul["GF"]
        x = GF(mul["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = mul["Y"][:,:]
        x[:,:] *= y
        assert np.all(x[:,:] == mul["Z"][:,:])

    def test_rdiv_scalar(self, div):
        GF = div["GF"]
        i = np.random.randint(0, div["Z"].shape[0])  # Random x index
        j = np.random.randint(0, div["Z"].shape[1])  # Random y index
        x = GF(div["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = div["Y"][i,j]
        x[i,j] /= y
        assert x[i,j] == div["Z"][i,j]

    def test_rdiv_vector(self, div):
        GF = div["GF"]
        i = np.random.randint(0, div["Z"].shape[0])  # Random x index
        x = GF(div["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = div["Y"][i,:]
        x[i,:] /= y
        assert np.all(x[i,:] == div["Z"][i,:])

    def test_rdiv_matrix(self, div):
        GF = div["GF"]
        x = GF(div["X"])  # Ensure this is a copy operation to avoid rewriting the LUT!
        y = div["Y"][:,:]
        x[:,:] /= y
        assert np.all(x[:,:] == div["Z"][:,:])


class TestArithmeticAssertions:
    def test_div_0(self, field):
        x = field.Random(field.order)
        y = 0
        with pytest.raises(AssertionError):
            z = x / y
        y = np.arange(0, field.order, dtype=int)
        with pytest.raises(AssertionError):
            z = x / y
        y = field.Elements()
        with pytest.raises(AssertionError):
            z = x / y

    def test_mul_inv_0(self, field):
        x = field.Elements()
        # with pytest.raises(AssertionError):
        #     z = 1 / x
        with pytest.raises(AssertionError):
            z = x ** -1

    def test_exp_0_negative_power(self, field):
        x = field.Elements()
        y = -3
        with pytest.raises(AssertionError):
            z = x ** y
        y = -3*np.ones(x.size, dtype=int)
        with pytest.raises(AssertionError):
            z = x ** y

    def test_log_0(self, field):
        x = field.Elements()
        with pytest.raises(AssertionError):
            z = np.log(x)
        with pytest.raises(AssertionError):
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
        with pytest.raises(AssertionError):
            c = a + b
        with pytest.raises(AssertionError):
            c = b + a

    def test_vector_int_out_of_range(self, field):
        shape = (10,)
        a = field.Random(shape)
        b = field.order
        with pytest.raises(AssertionError):
            c = a + b
        with pytest.raises(AssertionError):
            c = b + a

    def test_matrix_int_out_of_range(self, field):
        shape = (10,10)
        a = field.Random(shape)
        b = field.order
        with pytest.raises(AssertionError):
            c = a + b
        with pytest.raises(AssertionError):
            c = b + a

    def test_scalar_scalar_out_of_range(self, field):
        shape = ()
        a = field.Random(shape)
        b = field.order*np.ones(shape, dtype=int)
        with pytest.raises(AssertionError):
            c = a + b
        # TODO: Can't figure out how to make this fail
        # with pytest.raises(AssertionError):
        #     c = b + a

    def test_vector_vector_out_of_range(self, field):
        shape = (10,)
        a = field.Random(shape)
        b = field.order*np.ones(shape, dtype=int)
        with pytest.raises(AssertionError):
            c = a + b
        with pytest.raises(AssertionError):
            c = b + a

    def test_matrix_matrix_out_of_range(self, field):
        shape = (10,10)
        a = field.Random(shape)
        b = field.order*np.ones(shape, dtype=int)
        with pytest.raises(AssertionError):
            c = a + b
        with pytest.raises(AssertionError):
            c = b + a


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

    # def test_prim_poly(self, field):
    #     prim_poly = field.prim_poly  # Polynomial in GF(p)
    #     alpha = field.alpha
    #     poly = galois.Poly(prim_poly.coeffs, field=field)  # Polynomial in GF(p^m)
    #     assert poly(alpha) == 0

    def test_freshmans_dream(self, field):
        a = field.Random(10)
        b = field.Random(10)
        p = field.characteristic
        assert np.all((a + b)**p == a**p + b**p)

    def test_fermats_little_theorem(self, field):
        poly = galois.Poly([1], field=field)  # Base polynomial
        p = field.characteristic
        for a in field.Elements():
            poly = poly * galois.Poly([1, -a], field=field)
        assert poly == galois.Poly.Powers([1, -1], [p, 1], field=field)


def check_array(array, GF):
    assert type(array) is GF
    assert array.dtype == GF._dtype
