"""
A pytest module to test methods of Galois field array classes.
"""
import random

import numpy as np
import pytest

import galois


def test_display_method():
    GF = galois.GF(2**3)
    a = GF([1, 5, 2])
    assert str(a) == "GF([1, 5, 2], order=2^3)"
    GF.display("poly")
    assert str(a) == "GF([1, α^2 + 1, α], order=2^3)"
    GF.display()
    assert str(a) == "GF([1, 5, 2], order=2^3)"


def test_display_context_manager():
    GF = galois.GF(2**3)
    a = GF([1, 5, 2])
    assert str(a) == "GF([1, 5, 2], order=2^3)"
    with GF.display("poly"):
        assert str(a) == "GF([1, α^2 + 1, α], order=2^3)"
    assert str(a) == "GF([1, 5, 2], order=2^3)"


def test_display_poly_var_method():
    GF = galois.GF(2**3)
    a = GF([1, 5, 2])
    assert str(a) == "GF([1, 5, 2], order=2^3)"
    GF.display("poly", "x")
    assert str(a) == "GF([1, x^2 + 1, x], order=2^3)"
    GF.display()
    assert str(a) == "GF([1, 5, 2], order=2^3)"


def test_display_poly_var_context_manager():
    GF = galois.GF(2**3)
    a = GF([1, 5, 2])
    assert str(a) == "GF([1, 5, 2], order=2^3)"
    with GF.display("poly", "x"):
        assert str(a) == "GF([1, x^2 + 1, x], order=2^3)"
    assert str(a) == "GF([1, 5, 2], order=2^3)"
