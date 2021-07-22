"""
A pytest module to test the Galois LFSR implementation.
"""
import pytest
import numpy as np

import galois


def test_gf2_output_1():
    """
    The states of the Galois LFSR generate the binary extension field with the connection polynomial as its
    irreducible polynomial.
    """
    GF = galois.GF2
    poly = galois.conway_poly(2, 4)
    state = GF([0,0,0,1])
    lfsr = galois.LFSR(poly, state=state, config="galois")

    GFE = galois.GF(2**4, irreducible_poly=poly)
    alpha = GFE.primitive_element

    for i in range(GFE.order - 1):
        np.array_equal(lfsr.state, (alpha**i).vector())
        lfsr.step()


def test_gf2_output_2():
    """
    The states of the Galois LFSR generate the binary extension field with the connection polynomial as its
    irreducible polynomial.
    """
    GF = galois.GF2
    poly = galois.conway_poly(2, 8)
    state = GF([0,0,0,0,0,0,0,1])
    lfsr = galois.LFSR(poly, state=state, config="galois")

    GFE = galois.GF(2**8, irreducible_poly=poly)
    alpha = GFE.primitive_element

    for i in range(GFE.order - 1):
        np.array_equal(lfsr.state, (alpha**i).vector())
        lfsr.step()


def test_berlekamp_massey_gf2_1():
    """
    SageMath:
        F = GF(2)
        s = [0,0,1,1,0,1,1,1,0,1]
        s = [F(1) if si == 1 else F(0) for si in s]
        berlekamp_massey(s)
    """
    GF = galois.GF2
    s = GF([0,0,1,1,0,1,1,1,0,1])
    c_truth = galois.Poly.Degrees([5,2,0])
    c = galois.berlekamp_massey(s, config="galois")
    assert c == c_truth


def test_berlekamp_massey_gf2_2():
    """
    SageMath:
        F = GF(2)
        s = [1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,1,1,0]
        s = [F(1) if si == 1 else F(0) for si in s]
        berlekamp_massey(s)
    """
    GF = galois.GF2
    s = GF([1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,1,1,0])
    c_truth = galois.Poly.Degrees([8,6,5,4,0])
    c = galois.berlekamp_massey(s, config="galois")
    assert c == c_truth


def test_berlekamp_massey_gf2_3():
    """
    SageMath:
        F = GF(2)
        s = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,1,1,0,1,1]
        s = [F(1) if si == 1 else F(0) for si in s]
        berlekamp_massey(s)
    """
    GF = galois.GF2
    s = GF([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,1,1,0,1,1])
    c_truth = galois.Poly.Degrees([100,97,95,94,92,91,89,85,84,81,80,78,76,75,73,70,69,66,65,64,63,59,57,56,55,54,53,52,48,45,44,43,0])
    c = galois.berlekamp_massey(s, config="galois")
    assert c == c_truth
