"""
A pytest module to test the algebraic structure functions related to finite fields.
"""
import pytest

import galois


def test_not_abstract_classes():
    assert not galois.is_field(galois.array.Array)
    assert not galois.is_field(galois.FieldArray)

    assert not galois.is_prime_field(galois.array.Array)
    assert not galois.is_prime_field(galois.FieldArray)

    assert not galois.is_extension_field(galois.array.Array)
    assert not galois.is_extension_field(galois.FieldArray)


def test_not_metaclasses():
    assert not galois.is_field(galois.meta.Meta)
    assert not galois.is_field(galois.FieldMeta)

    assert not galois.is_prime_field(galois.meta.Meta)
    assert not galois.is_prime_field(galois.FieldMeta)

    assert not galois.is_extension_field(galois.meta.Meta)
    assert not galois.is_extension_field(galois.FieldMeta)


def test_array_classes():
    GF = galois.GF(2)
    assert galois.is_field(GF)
    assert galois.is_prime_field(GF)
    assert not galois.is_extension_field(GF)

    GF = galois.GF(7)
    assert galois.is_field(GF)
    assert galois.is_prime_field(GF)
    assert not galois.is_extension_field(GF)

    GF = galois.GF(2**8)
    assert galois.is_field(GF)
    assert not galois.is_prime_field(GF)
    assert galois.is_extension_field(GF)

    GF = galois.GF(7**3)
    assert galois.is_field(GF)
    assert not galois.is_prime_field(GF)
    assert galois.is_extension_field(GF)


def test_array_instances():
    GF = galois.GF(2)
    a = GF.Random(10)
    assert galois.is_field(a)
    assert galois.is_prime_field(a)
    assert not galois.is_extension_field(a)

    GF = galois.GF(7)
    a = GF.Random(10)
    assert galois.is_field(a)
    assert galois.is_prime_field(a)
    assert not galois.is_extension_field(a)

    GF = galois.GF(2**8)
    a = GF.Random(10)
    assert galois.is_field(a)
    assert not galois.is_prime_field(a)
    assert galois.is_extension_field(a)

    GF = galois.GF(7**3)
    a = GF.Random(10)
    assert galois.is_field(a)
    assert not galois.is_prime_field(a)
    assert galois.is_extension_field(a)
