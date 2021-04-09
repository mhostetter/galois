"""
A pytest module to test Galois field polynomial instantiation.
"""
import random

import numpy as np
import pytest

import galois


FIELDS = [
    galois.GF2,  # GF(2)
    galois.GF(31),  # GF(p) with np.int dtypes
    galois.GF(36893488147419103183),  # GF(p) with object dtype
    galois.GF(2**8),  # GF(2^m) with np.int dtypes
    galois.GF(2**100),  # GF(2^m) with object dtype
]


@pytest.fixture(params=FIELDS)
def config(field):
    d = {}
    d["GF"] = field
    d["degree"] = 5

    c1, c2, c3 = field.Random(low=1), field.Random(low=1), field.Random(low=1)
    d["nonzero_degrees"] = [5, 3, 0]
    d["nonzero_coeffs"] = [c1, c2, c3]

    d["degrees"] = [5, 4, 3, 2, 1, 0]
    d["coeffs"] = [c1, 0, c2, 0, 0, c3]
    d["integer"] = int(c1)*field.order**5 + int(c2)*field.order**3 + int(c3)*field.order**0

    # Negative coefficients
    n1, n2, n3 = -abs(int(-c1)), -abs(int(-c2)), -abs(int(-c3))
    d["neg_coeffs"] = [n1, 0, n2, 0, 0, n3]
    d["neg_nonzero_coeffs"] = [n1, n2, n3]

    # Leading zeros
    d["lz_coeffs"] = [0, 0, c1, 0, c2, 0, 0, c3]

    # Mixed zeros
    d["mz_nonzero_degrees"] = [7, 6, 5, 3, 2, 0]
    d["mz_nonzero_coeffs"] = [0, 0, c1, c2, 0, c3]

    return d


class TestPoly:
    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_coeffs(self, type1, config):
        GF = config["GF"]
        if type1 is not galois.GFArray:
            p = galois.Poly(type1(config["coeffs"]), field=GF)
        else:
            p = galois.Poly(GF(config["coeffs"]))
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_leading_zero_coeffs(self, type1, config):
        GF = config["GF"]
        if type1 is not galois.GFArray:
            p = galois.Poly(type1(config["lz_coeffs"]), field=GF)
        else:
            p = galois.Poly(GF(config["lz_coeffs"]))
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_ascending_coeffs(self, type1, config):
        GF = config["GF"]
        if type1 is not galois.GFArray:
            p = galois.Poly(type1(config["coeffs"][::-1]), field=GF, order="asc")
        else:
            p = galois.Poly(GF(config["coeffs"][::-1]), order="asc")
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_negative_coeffs(self, type1, config):
        GF = config["GF"]
        p = galois.Poly(type1(config["neg_coeffs"]), field=GF)
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    @pytest.mark.parametrize("field", FIELDS)
    def test_zero(self, type1, field):
        # The zero polynomial can cause problems, so test it explicitly
        if type1 is not galois.GFArray:
            p = galois.Poly(type1([0]), field=field)
        else:
            p = galois.Poly(field([0]))
        assert isinstance(p, galois.Poly)
        assert p.field is field
        assert p.degree == 0
        assert np.array_equal(p.nonzero_degrees, [])
        assert np.array_equal(p.nonzero_coeffs, [])
        assert np.array_equal(p.degrees, [0])
        assert np.array_equal(p.coeffs, [0])
        assert p.integer == 0

    def check_attributes(self, poly, config):
        assert isinstance(poly, galois.Poly)
        assert poly.field is config["GF"]
        assert poly.degree == config["degree"]
        assert np.array_equal(poly.nonzero_degrees, config["nonzero_degrees"])
        assert np.array_equal(poly.nonzero_coeffs, config["nonzero_coeffs"])
        assert np.array_equal(poly.degrees, config["degrees"])
        assert np.array_equal(poly.coeffs, config["coeffs"])
        assert poly.integer == config["integer"]


class TestDensePoly:
    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_coeffs(self, type1, config):
        GF = config["GF"]
        if type1 is not galois.GFArray:
            p = galois.poly.DensePoly(type1(config["coeffs"]), field=GF)
        else:
            p = galois.poly.DensePoly(GF(config["coeffs"]))
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_leading_zero_coeffs(self, type1, config):
        GF = config["GF"]
        if type1 is not galois.GFArray:
            p = galois.poly.DensePoly(type1(config["lz_coeffs"]), field=GF)
        else:
            p = galois.poly.DensePoly(GF(config["lz_coeffs"]))
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    def test_ascending_coeffs(self, type1, config):
        GF = config["GF"]
        if type1 is not galois.GFArray:
            p = galois.poly.DensePoly(type1(config["coeffs"][::-1]), field=GF, order="asc")
        else:
            p = galois.poly.DensePoly(GF(config["coeffs"][::-1]), order="asc")
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    def test_negative_coeffs(self, type1, config):
        GF = config["GF"]
        p = galois.poly.DensePoly(type1(config["neg_coeffs"]), field=GF)
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array, galois.GFArray])
    @pytest.mark.parametrize("field", FIELDS)
    def test_zero(self, type1, field):
        # The zero polynomial can cause problems, so test it explicitly
        if type1 is not galois.GFArray:
            p = galois.poly.DensePoly(type1([0]), field=field)
        else:
            p = galois.poly.DensePoly(field([0]))
        assert type(p) is galois.poly.DensePoly
        assert p.field is field
        assert p.degree == 0
        assert np.array_equal(p.nonzero_degrees, [])
        assert np.array_equal(p.nonzero_coeffs, [])
        assert np.array_equal(p.degrees, [0])
        assert np.array_equal(p.coeffs, [0])
        assert p.integer == 0

    def check_attributes(self, poly, config):
        assert type(poly) is galois.poly.DensePoly
        assert poly.field is config["GF"]
        assert poly.degree == config["degree"]
        assert np.array_equal(poly.nonzero_degrees, config["nonzero_degrees"])
        assert np.array_equal(poly.nonzero_coeffs, config["nonzero_coeffs"])
        assert np.array_equal(poly.degrees, config["degrees"])
        assert np.array_equal(poly.coeffs, config["coeffs"])
        assert poly.integer == config["integer"]


class TestSparsePoly:
    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    @pytest.mark.parametrize("type2", [list, tuple, np.array, galois.GFArray])
    def test_coeffs(self, type1, type2, config):
        GF = config["GF"]
        if type2 is not galois.GFArray:
            p = galois.poly.SparsePoly(type1(config["nonzero_degrees"]), type2(config["nonzero_coeffs"]), field=GF)
        else:
            p = galois.poly.SparsePoly(type1(config["nonzero_degrees"]), GF(config["nonzero_coeffs"]))
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    @pytest.mark.parametrize("type2", [list, tuple, np.array, galois.GFArray])
    def test_zeros_coeffs(self, type1, type2, config):
        GF = config["GF"]
        if type2 is not galois.GFArray:
            p = galois.poly.SparsePoly(type1(config["mz_nonzero_degrees"]), type2(config["mz_nonzero_coeffs"]), field=GF)
        else:
            p = galois.poly.SparsePoly(type1(config["mz_nonzero_degrees"]), GF(config["mz_nonzero_coeffs"]))
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    @pytest.mark.parametrize("type2", [list, tuple, np.array])
    def test_negative_coeffs(self, type1, type2, config):
        GF = config["GF"]
        p = galois.poly.SparsePoly(type1(config["nonzero_degrees"]), type2(config["neg_nonzero_coeffs"]), field=GF)
        self.check_attributes(p, config)

    @pytest.mark.parametrize("type1", [list, tuple, np.array])
    @pytest.mark.parametrize("type2", [list, tuple, np.array, galois.GFArray])
    @pytest.mark.parametrize("field", FIELDS)
    def test_zero(self, type1, type2, field):
        # The zero polynomial can cause problems, so test it explicitly
        if type2 is not galois.GFArray:
            p = galois.poly.SparsePoly(type1([0]), type2([0]), field=field)
        else:
            p = galois.poly.SparsePoly(type1([0]), field([0]))
        assert type(p) is galois.poly.SparsePoly
        assert p.field is field
        assert p.degree == 0
        assert np.array_equal(p.nonzero_degrees, [])
        assert np.array_equal(p.nonzero_coeffs, [])
        assert np.array_equal(p.degrees, [0])
        assert np.array_equal(p.coeffs, [0])
        assert p.integer == 0

    def check_attributes(self, poly, config):
        assert type(poly) is galois.poly.SparsePoly
        assert poly.field is config["GF"]
        assert poly.degree == config["degree"]
        assert np.array_equal(poly.nonzero_degrees, config["nonzero_degrees"])
        assert np.array_equal(poly.nonzero_coeffs, config["nonzero_coeffs"])
        assert np.array_equal(poly.degrees, config["degrees"])
        assert np.array_equal(poly.coeffs, config["coeffs"])
        assert poly.integer == config["integer"]
