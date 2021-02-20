import pytest

import galois


def test_example():
    a = 1
    b = galois.example_function(a)
    assert a == b
