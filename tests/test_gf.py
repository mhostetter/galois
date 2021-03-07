"""
A pytest module to test classes/functions in galois/gf.py.
"""
import pytest

import galois


def test_cant_instantiate_GFBase():
    with pytest.raises(NotImplementedError):
        a = galois.gf.GFBase([1,2,3])


def test_cant_instantiate_GFpBase():
    with pytest.raises(NotImplementedError):
        a = galois.GFpBase([1,2,3])
