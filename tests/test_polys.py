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
        with pytest.raises(AssertionError):
            p.coeffs = l2

    def test_equal(self, field):
        c = field.Random(6)
        c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        p1 = galois.Poly(c)
        p2 = galois.Poly(c.tolist(), field=field)
        assert p1 == p2


class TestArithmetic:
    def test_add(self, poly_add):
        for i in range(len(poly_add["X"])):
            x = poly_add["X"][i]
            y = poly_add["Y"][i]
            z = x + y
            assert z == poly_add["Z"][i]
            check_poly(z, poly_add["GF"])

    def test_subtract(self, poly_subtract):
        for i in range(len(poly_subtract["X"])):
            x = poly_subtract["X"][i]
            y = poly_subtract["Y"][i]
            z = x - y
            assert z == poly_subtract["Z"][i]
            check_poly(z, poly_subtract["GF"])

    def test_multiply(self, poly_multiply):
        for i in range(len(poly_multiply["X"])):
            x = poly_multiply["X"][i]
            y = poly_multiply["Y"][i]
            z = x * y
            assert z == poly_multiply["Z"][i]
            check_poly(z, poly_multiply["GF"])

    def test_divmod(self, poly_divmod):
        for i in range(len(poly_divmod["X"])):
            x = poly_divmod["X"][i]
            y = poly_divmod["Y"][i]
            q = x // y
            r = x % y
            assert q == poly_divmod["Q"][i]
            assert r == poly_divmod["R"][i]
            check_poly(q, poly_divmod["GF"])
            check_poly(r, poly_divmod["GF"])

    def test_power(self, poly_power):
        x = poly_power["X"]  # Single polynomial
        for i in range(len(poly_power["Y"])):
            y = poly_power["Y"][i]
            z = x ** y
            assert z == poly_power["Z"][i]
            check_poly(z, poly_power["GF"])

    def test_evaluate_constant(self, poly_evaluate):
        for i in range(len(poly_evaluate["X"])):
            for j in range(poly_evaluate["Y"].size):
                x = poly_evaluate["X"][i]  # Polynomial
                y = poly_evaluate["Y"][j]  # GF element
                z = x(y)  # GF element
                assert z == poly_evaluate["Z"][i,j]

    def test_evaluate_vector(self, poly_evaluate):
        for i in range(len(poly_evaluate["X"])):
            x = poly_evaluate["X"][i]  # Polynomial
            y = poly_evaluate["Y"]  # GF array
            z = x(y)  # GF array
            assert np.all(z == poly_evaluate["Z"][i,:])


class TestArithmeticTypes:
    def test_evaluate_int_scalar(self, poly_evaluate):
        pass

    def test_evaluate_int_arrary(self, poly_evaluate):
        pass

    def test_evaluate_field_scalar(self, poly_evaluate):
        pass

    def test_evaluate_field_array(self, poly_evaluate):
        pass


def check_poly(poly, GF):
    assert isinstance(poly, galois.Poly)
    assert poly.field is GF
    assert type(poly.coeffs) is GF
