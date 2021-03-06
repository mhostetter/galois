"""
A pytest module to test classes/functions in galois/gf.py.
"""
import pytest

import galois


def test_cant_instantiateGFBase():
    with pytest.raises(AssertionError):
        a = galois.gf.GFBase([1,2,3])


def test_cant_instantiateGFBasep():
    with pytest.raises(AssertionError):
        a = galois.GFpBase([1,2,3])
