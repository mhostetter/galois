"""
A pytest module to test Galois field polynomials.
"""
import pytest
import numpy as np

import galois


class TestInstantiation:
    def test_field_coeffs(self, field):
        c = field.Random(6)
        c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        p = galois.Poly(c)
        assert np.all(p.coeffs == c)
        check_poly(p, field)

    def test_list_coeffs(self, field):
        c = field.Random(6)
        c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        l = c.tolist()
        p = galois.Poly(l, field=field)
        assert np.all(p.coeffs == c)
        check_poly(p, field)

    def test_field_leading_zero_coeffs(self, field):
        c = field.Random(6)
        c[0] = 0  # Ensure leading coefficient is zero
        c[1] = field.Random(low=1)  # Ensure next leading coefficient is non-zero
        p = galois.Poly(c)
        assert np.all(p.coeffs == c[1:])
        check_poly(p, field)

    def test_list_leading_zero_coeffs(self, field):
        c = field.Random(6)
        c[0] = 0  # Ensure leading coefficient is zero
        c[1] = field.Random(low=1)  # Ensure next leading coefficient is non-zero
        l = c.tolist()
        p = galois.Poly(l, field=field)
        assert np.all(p.coeffs == c[1:])
        check_poly(p, field)

    def test_field_coeffs_asc_order(self, field):
        c1 = field.Random(6)
        c1[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        p1 = galois.Poly(c1)
        assert np.all(p1.coeffs == c1)
        check_poly(p1, field)

        c2 = np.flip(c1)
        p2 = galois.Poly(c2, order="asc")
        assert np.all(p2.coeffs == c2)
        check_poly(p2, field)

        assert p1 == p2

    # def test_list_coeffs(self, field):
    #     c = field.Random(6)
    #     c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    #     l = c.tolist()
    #     p = galois.Poly(l, field=field)
    #     assert np.all(p.coeffs == c)
    #     check_poly(p, field)

    def test_negative_coeffs(self, field):
        a = field.Random()
        l = [1, -int(a)]
        p = galois.Poly(l, field=field)
        assert np.all(p.coeffs == [1, -a])
        check_poly(p, field)

    def test_update_coeffs_field(self, field):
        c = field.Random(6)
        c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        p = galois.Poly(c)
        assert np.all(p.coeffs == c)
        check_poly(p, field)

        c2 = field.Random(3)
        c2[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        p.coeffs = c2
        assert np.all(p.coeffs == c2)
        check_poly(p, field)

    def test_update_coeffs_list(self, field):
        c = field.Random(6)
        c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        l = c.tolist()
        p = galois.Poly(l, field=field)
        assert np.all(p.coeffs == c)
        check_poly(p, field)

        c2 = field.Random(3)
        c2[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        l2 = c2.tolist()
        with pytest.raises(TypeError):
            p.coeffs = l2

    def test_equal(self, field):
        c = field.Random(6)
        c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        p1 = galois.Poly(c)
        p2 = galois.Poly(c.tolist(), field=field)
        assert p1 == p2


class TestArithmetic:
    def test_add(self, poly_add):
        GF, X, Y, Z = poly_add["GF"], poly_add["X"], poly_add["Y"], poly_add["Z"]
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            z = x + y
            assert z == Z[i]
            check_poly(z, GF)

    def test_subtract(self, poly_subtract):
        GF, X, Y, Z = poly_subtract["GF"], poly_subtract["X"], poly_subtract["Y"], poly_subtract["Z"]
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            z = x - y
            assert z == Z[i]
            check_poly(z, GF)

    def test_multiply(self, poly_multiply):
        GF, X, Y, Z = poly_multiply["GF"], poly_multiply["X"], poly_multiply["Y"], poly_multiply["Z"]
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            z = x * y
            assert z == Z[i]
            check_poly(z, GF)

    def test_divmod(self, poly_divmod):
        GF, X, Y, Q, R = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["Q"], poly_divmod["R"]
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            q = x // y
            r = x % y
            assert q == Q[i]
            assert r == R[i]
            check_poly(q, GF)
            check_poly(r, GF)

    def test_power(self, poly_power):
        GF, X, Y, Z = poly_power["GF"], poly_power["X"], poly_power["Y"], poly_power["Z"]
        x = X  # Single polynomial
        for i in range(len(Y)):
            y = Y[i]
            z = x ** y
            assert z == Z[i]
            check_poly(z, GF)

    def test_evaluate_constant(self, poly_evaluate):
        GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
        for i in range(len(X)):
            for j in range(Y.size):
                x = X[i]  # Polynomial
                y = Y[j]  # GF element
                z = x(y)  # GF element
                assert z == Z[i,j]

    def test_evaluate_vector(self, poly_evaluate):
        GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
        for i in range(len(X)):
            x = X[i]  # Polynomial
            y = Y  # GF array
            z = x(y)  # GF array
            assert np.all(z == Z[i,:])


class TestArithmeticTypes:
    def test_evaluate_int_scalar(self, poly_evaluate):
        pass

    def test_evaluate_int_arrary(self, poly_evaluate):
        pass

    def test_evaluate_field_scalar(self, poly_evaluate):
        pass

    def test_evaluate_field_array(self, poly_evaluate):
        pass


class TestProperties:
    def test_decimal(self):
        poly = galois.Poly([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,0,0,1])
        assert poly.decimal == 4295000729

        poly = galois.Poly.NonZero([1,1,1,1,1,1,1], [32,15,9,7,4,3,0])
        assert poly.decimal == 4295000729


def check_poly(poly, GF):
    assert isinstance(poly, galois.Poly)
    assert poly.field is GF
    assert type(poly.coeffs) is GF
