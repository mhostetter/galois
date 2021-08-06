"""
A pytest module to test the functions in math_.py.
"""
import pytest

import galois


def test_prod():
    assert galois.prod([2, 4, 14]) == 2*4*14
    assert galois.prod([2, 4, 14], start=2) == 2*2*4*14
