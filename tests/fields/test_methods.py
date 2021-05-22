"""
A pytest module to test methods of Galois field array classes.
"""
import pytest
import numpy as np

import galois


def test_display_method():
    GF = galois.GF(2**3)
    a = GF([1, 0, 5, 2])
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"
    GF.display("poly")
    assert str(a) == "GF([1, 0, α^2 + 1, α], order=2^3)"
    GF.display("power")
    assert str(a) == "GF([1, 0, α^6, α], order=2^3)"
    GF.display()
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"


def test_display_context_manager():
    GF = galois.GF(2**3)
    a = GF([1, 0, 5, 2])
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"
    with GF.display("poly"):
        assert str(a) == "GF([1, 0, α^2 + 1, α], order=2^3)"
    with GF.display("power"):
        assert str(a) == "GF([1, 0, α^6, α], order=2^3)"
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"


def test_display_exceptions():
    GF = galois.GF(2**3)
    a = GF([1, 0, 5, 2])
    with pytest.raises(ValueError):
        GF.display("invalid-display-type")
