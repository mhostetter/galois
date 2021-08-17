"""
A pytest module to test the functions in _math.py.
"""
import galois


def test_prod():
    assert galois.prod(2, 4, 14) == 2*4*14
