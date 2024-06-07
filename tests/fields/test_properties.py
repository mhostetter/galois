"""
A pytest module to test various Galois field properties.
"""

import random

import numpy as np

import galois


def test_properties(field_properties):
    GF = field_properties["GF"]
    assert GF.characteristic == field_properties["characteristic"]
    assert GF.degree == field_properties["degree"]
    assert GF.order == field_properties["order"]
    assert GF.primitive_element == field_properties["primitive_element"]
    assert GF.irreducible_poly == field_properties["irreducible_poly"]


def test_characteristic(field):
    if field.order < 2**16:
        a = field.elements
    else:
        # Only select some, not all, elements for very large fields
        a = field.Random(2**16)
    p = field.characteristic
    b = a * p
    assert np.all(b == 0)


def test_element_order(field):
    if field.order > 1e6:  # TODO: Skip for extremely large fields
        return
    if field.order < 2**16:
        a = field.units
    else:
        # Only select some, not all, elements for very large fields
        a = field.Random(2**16, low=1)
    q = field.order
    assert np.all(a ** (q - 1) == 1)


def test_primitive_element_is_generator(field):
    if field.order > 1e6:  # TODO: Skip for extremely large fields
        return
    a = random.choice(field.primitive_elements)
    elements = a ** np.arange(0, field.order - 1)
    assert len(set(elements.tolist())) == field.order - 1


def test_primitive_root_of_unity():
    GF = galois.GF(3**5)
    i, j = 0, 0
    while i < 5 and j < 5:
        n = random.randint(2, GF.order - 1)
        if (GF.order - 1) % n == 0 and i < 5:
            r = GF.primitive_root_of_unity(n)
            assert not np.any(np.power.outer(r, np.arange(1, n)) == 1)
            assert np.all(r**n == 1)
            i += 1
        elif j < 5:
            x = GF.elements
            assert np.any(x**n == 1)

    # Large field
    GF = galois.GF(2**100)
    i = 0
    while i < 5:
        n = random.randint(2, 1000)
        if (GF.order - 1) % n == 0 and i < 5:
            r = GF.primitive_root_of_unity(n)
            assert not np.any(np.power.outer(r, np.arange(1, n)) == 1)
            assert np.all(r**n == 1)
            i += 1


def test_primitive_roots_of_unity():
    GF = galois.GF(3**5)
    i, j = 0, 0
    while i < 5 and j < 5:
        n = random.randint(2, GF.order - 1)
        if (GF.order - 1) % n == 0 and i < 5:
            r = GF.primitive_roots_of_unity(n)
            assert not np.any(np.power.outer(r, np.arange(1, n)) == 1)
            assert np.all(r**n == 1)
            i += 1
        elif j < 5:
            x = GF.elements
            assert np.any(x**n == 1)


def test_irreducible_poly(field):
    poly = field.irreducible_poly  # Polynomial in GF(p)
    alpha = field.primitive_element
    assert field.is_primitive_poly == (poly(alpha, field=field) == 0)


def test_freshmans_dream(field):
    a = field.Random(10)
    b = field.Random(10)
    p = field.characteristic
    assert np.all((a + b) ** p == a**p + b**p)


def test_fermats_little_theorem(field):
    if field.order > 50:
        # Skip for very large fields because this takes too long
        return
    poly = galois.Poly([1], field=field)  # Base polynomial
    # p = field.characteristic
    for a in field.elements:
        poly = poly * galois.Poly([1, -a], field=field)
    assert poly == galois.Poly.Degrees([field.order, 1], coeffs=[1, -1], field=field)


def test_exp_log_duality(field):
    if field.order > 2**16:  # TODO: Skip slow log() for very large fields
        return
    alpha = field.primitive_element
    x = field.Random(10, low=1)
    e = np.log(x)
    assert np.all(alpha**e == x)
