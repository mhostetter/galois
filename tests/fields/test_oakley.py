"""
A pytest module to test the Oakley fields.
"""
import pytest

import galois


def test_oakley_1():
    GF = galois.Oakley1()
    assert issubclass(GF, galois.FieldArray)
    assert isinstance(GF, galois.FieldClass)
    assert GF.is_prime_field


def test_oakley_2():
    GF = galois.Oakley2()
    assert issubclass(GF, galois.FieldArray)
    assert isinstance(GF, galois.FieldClass)
    assert GF.is_prime_field


def test_oakley_3():
    GF = galois.Oakley3()
    assert issubclass(GF, galois.FieldArray)
    assert issubclass(type(GF), galois.FieldClass)
    assert GF.is_extension_field


def test_oakley_4():
    GF = galois.Oakley4()
    assert issubclass(GF, galois.FieldArray)
    assert issubclass(type(GF), galois.FieldClass)
    assert GF.is_extension_field
