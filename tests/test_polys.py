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

    def test_sub(self, poly_sub):
        for i in range(len(poly_sub["X"])):
            x = poly_sub["X"][i]
            y = poly_sub["Y"][i]
            z = x - y
            assert z == poly_sub["Z"][i]
            check_poly(z, poly_sub["GF"])

    def test_mul(self, poly_mul):
        for i in range(len(poly_mul["X"])):
            x = poly_mul["X"][i]
            y = poly_mul["Y"][i]
            z = x * y
            assert z == poly_mul["Z"][i]
            check_poly(z, poly_mul["GF"])

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

    def test_exp(self, poly_exp):
        x = poly_exp["X"]  # Single polynomial
        for i in range(len(poly_exp["Y"])):
            y = poly_exp["Y"][i]
            z = x ** y
            assert z == poly_exp["Z"][i]
            check_poly(z, poly_exp["GF"])

    def test_eval_constant(self, poly_eval):
        for i in range(len(poly_eval["X"])):
            for j in range(poly_eval["Y"].size):
                x = poly_eval["X"][i]  # Polynomial
                y = poly_eval["Y"][j]  # GF element
                z = x(y)  # GF element
                assert z == poly_eval["Z"][i,j]

    def test_eval_vector(self, poly_eval):
        for i in range(len(poly_eval["X"])):
            x = poly_eval["X"][i]  # Polynomial
            y = poly_eval["Y"]  # GF array
            z = x(y)  # GF array
            assert np.all(z == poly_eval["Z"][i,:])


class TestArithmeticTypes:
    def test_eval_int_scalar(self, poly_eval):
        pass

    def test_eval_int_arrary(self, poly_eval):
        pass

    def test_eval_field_scalar(self, poly_eval):
        pass

    def test_eval_field_array(self, poly_eval):
        pass


def check_poly(poly, GF):
    assert isinstance(poly, galois.Poly)
    assert poly.field is GF
    assert type(poly.coeffs) is GF
