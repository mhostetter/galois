"""
A pytest module to test various Galois field properties.
"""
import numpy as np
import pytest

import galois


def test_characteristic(field):
    if field.order < 2**16:
        a = field.Elements()
    else:
        # Only select some, not all, elements for very large fields
        a = field.Random(2**16)
    p = field.characteristic
    b = a * p
    assert np.all(b == 0)


def test_property_2(field):
    if field.order > 1e6:  # TODO: Skip for extremely large fields
        return
    if field.order < 2**16:
        a = field.Elements()[1:]
    else:
        # Only select some, not all, elements for very large fields
        a = field.Random(2**16, low=1)
    q = field.order
    assert np.all(a**q == a)


def test_irreducible_poly(field):
    prim_poly = field.irreducible_poly  # Polynomial in GF(p)
    alpha = field.primitive_element
    poly = galois.Poly(prim_poly.coeffs, field=field)  # Polynomial in GF(p^m)
    assert poly(alpha) == 0


def test_freshmans_dream(field):
    a = field.Random(10)
    b = field.Random(10)
    p = field.characteristic
    assert np.all((a + b)**p == a**p + b**p)


def test_fermats_little_theorem(field):
    if field.order > 50:
        # Skip for very large fields because this takes too long
        return
    poly = galois.Poly([1], field=field)  # Base polynomial
    # p = field.characteristic
    for a in field.Elements():
        poly = poly * galois.Poly([1, -a], field=field)
    assert poly == galois.Poly.Degrees([field.order, 1], coeffs=[1, -1], field=field)


def test_exp_log_duality(field):
    if field.order > 2**16:  # TODO: Skip slow log() for very large fields
        return
    alpha = field.primitive_element
    x = field.Random(10, low=1)
    e = np.log(x)
    assert np.all(alpha**e == x)
