"""
A pytest module to test classes/functions in galois/gf.py.
"""
import pytest

import galois


def test_cant_instantiate_GF():
    with pytest.raises(AssertionError):
        a = galois.gf._GF([1,2,3])


def test_cant_instantiate_GFp():
    with pytest.raises(AssertionError):
        a = galois.GFp([1,2,3])
