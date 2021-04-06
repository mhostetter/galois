"""
A pytest module to test Galois field polynomial instantiation.
"""
import random

import numpy as np
import pytest

import galois


class TestField:
    def test_non_zero_leading_coeff(self, field):
        c = field.Random(6)
        c[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        p = galois.Poly(c)
        assert np.all(p.coeffs == c)
        assert isinstance(p, galois.Poly)
        assert p.field is field
        assert type(p.coeffs) is field

    def test_zero_leading_coeff(self, field):
        c = field.Random(6)
        c[0] = 0  # Ensure leading coefficient is zero
        c[1] = field.Random(low=1)  # Ensure next leading coefficient is non-zero
        p = galois.Poly(c)
        assert np.all(p.coeffs == c[1:])
        assert isinstance(p, galois.Poly)
        assert p.field is field
        assert type(p.coeffs) is field

    def test_overridding_field_argument(self, field):
        if field is not galois.GF2:
            c = field([1, 0, 1, 1])
            p = galois.Poly(c, field=galois.GF2)
            assert isinstance(p, galois.Poly)
            assert p.field is galois.GF2
            assert type(p.coeffs) is galois.GF2

    def test_invalid_field_argument(self, field):
        c = field([1, 0, 1, 1])
        with pytest.raises(TypeError):
            p = galois.Poly(c, field=2)

    # def test_coeffs_asc_order(self, field):
    #     c1 = field.Random(6)
    #     c1[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
    #     p1 = galois.Poly(c1)
    #     assert np.all(p1.coeffs == c1)
    #     check_poly(p1, field)

    #     c2 = np.flip(c1)
    #     p2 = galois.Poly(c2, order="asc")
    #     assert np.all(p2.coeffs == c2)
    #     check_poly(p2, field)

    #     assert p1 == p2


class TestList:
    def test_list_without_field(self, field):
        c = [random.randint(0, field.order - 1) for _ in range(6)]
        with pytest.raises(TypeError):
            p = galois.Poly(c, field=field.order)  # Passing in the field's order, not the field

    def test_non_zero_leading_coeff(self, field):
        c = [random.randint(0, field.order - 1) for _ in range(6)]
        c[0] = random.randint(1, field.order - 1)  # Ensure leading coefficient is non-zero
        p = galois.Poly(c, field=field)
        # assert np.all(p.coeffs == c)  # TODO: This sometimes fails for really large ints and dtype=object, below is a workaround
        assert p.coeffs.size == len(c) and all(p.coeffs[i] == c[i] for i in range(p.coeffs.size))
        assert isinstance(p, galois.Poly)
        assert p.field is field
        assert type(p.coeffs) is field

    def test_zero_leading_coeff(self, field):
        c = [random.randint(0, field.order - 1) for _ in range(6)]
        c[0] = 0  # Ensure leading coefficient is zero
        c[1] = random.randint(1, field.order - 1)  # Ensure next leading coefficient is non-zero
        p = galois.Poly(c, field=field)
        # assert np.all(p.coeffs == c[1:])
        assert p.coeffs.size == len(c[1:]) and all(p.coeffs[i] == c[1+i] for i in range(p.coeffs.size))
        assert isinstance(p, galois.Poly)
        assert p.field is field
        assert type(p.coeffs) is field

    def test_negative_coeffs(self, field):
        # Coefficients are +/- field elements with - indicating subtraction
        c = [random.randint(-(field.order - 1), field.order - 1) for _ in range(6)]
        c[0] = random.randint(1, field.order - 1)  # Ensure leading coefficient is positive, TODO: is this needed?
        p = galois.Poly(c, field=field)
        c_field = [field(e) if e > 0 else -field(abs(e)) for e in c]  # Convert +/- field elements into the field
        # assert np.all(p.coeffs == c_field)
        assert p.coeffs.size == len(c_field) and all(p.coeffs[i] == c_field[i] for i in range(p.coeffs.size))
        assert isinstance(p, galois.Poly)
        assert p.field is field
        assert type(p.coeffs) is field


class TestConstructors:
    def test_zero(self, field):
        p = galois.Poly.Zero(field=field)
        coeffs = field([0])
        assert np.array_equal(p.coeffs, coeffs)
        assert type(p) is galois.Poly
        assert p.field is field

    def test_one(self, field):
        p = galois.Poly.One(field=field)
        coeffs = field([1])
        assert np.array_equal(p.coeffs, coeffs)
        assert type(p) is galois.Poly
        assert p.field is field

    def test_identity(self, field):
        p = galois.Poly.Identity(field=field)
        coeffs = field([1, 0])
        assert np.array_equal(p.coeffs, coeffs)
        assert type(p) is galois.Poly
        assert p.field is field

    def test_random(self, field):
        degree = random.randint(0, 5)
        p = galois.Poly.Random(degree, field=field)
        assert type(p) is galois.Poly
        assert p.field is field
        assert p.degree == degree

    def test_integer(self, field):
        d = field.order + 1  # x + 1
        p = galois.Poly.Integer(d, field=field)
        coeffs = field([1, 1])
        assert np.array_equal(p.coeffs, coeffs)
        assert type(p) is galois.Poly
        assert p.field is field
        assert p.integer == d


    def test_integer_non_integer(self, field):
        d = field.order + 1  # x + 1
        d = float(d)
        with pytest.raises(TypeError):
            p = galois.Poly.Integer(d, field=field)

    def test_degrees(self, field):
        coeffs = [1, 0, 1, 0, 0, 1]
        p = galois.Poly.Degrees([5,3,0], field=field)
        assert np.array_equal(p.coeffs, coeffs)
        assert type(p) is galois.Poly
        assert p.field is field

    def test_roots_field(self, field):
        roots = field.Random(4)
        p = galois.Poly.Roots(roots, field=field)
        assert type(p) is galois.Poly
        assert p.field is field

        z = p(roots)
        assert np.all(z == 0)

    def test_roots_list(self, field):
        roots = field.Random(4).tolist()
        p = galois.Poly.Roots(roots, field=field)
        assert type(p) is galois.Poly
        assert p.field is field

        z = p(roots)
        assert np.all(z == 0)

    def test_roots_invalid_type(self, field):
        with pytest.raises(TypeError):
            p = galois.Poly.Roots(1.0, field=field)
        with pytest.raises(TypeError):
            p = galois.Poly.Roots([0, 1.0], field=field)
