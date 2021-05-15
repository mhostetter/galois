"""
A pytest module to test the algebraic structure functions related to finite groups.
"""
import pytest

import galois


def test_not_abstract_classes():
    assert not galois.is_group(galois.array.Array)
    assert not galois.is_group(galois.GroupArray)


def test_not_metaclasses():
    assert not galois.is_group(galois.meta.Meta)
    assert not galois.is_group(galois.GroupMeta)


def test_array_classes():
    G = galois.Group(16, "+")
    assert galois.is_group(G)

    G = galois.Group(16, "*")
    assert galois.is_group(G)


def test_array_instances():
    G = galois.Group(16, "+")
    a = G.Random(10)
    assert galois.is_group(a)

    G = galois.Group(16, "*")
    a = G.Random(10)
    assert galois.is_group(a)
