"""
A pytest module to test the Galois LFSR implementation.
"""
import pytest
import numpy as np

import galois


def test_exceptions():
    poly = galois.Poly.Degrees([7,1,0])

    with pytest.raises(TypeError):
        galois.LFSR(poly.coeffs)
    with pytest.raises(TypeError):
        galois.LFSR(poly, state=float(poly.integer))
    with pytest.raises(TypeError):
        galois.LFSR(poly.coeffs, config=1)

    with pytest.raises(ValueError):
        galois.LFSR(poly, config="invalid-argument")


def test_state():
    poly = galois.Poly.Degrees([7,1,0])

    lfsr = galois.LFSR(poly, state=1, config="galois")
    assert np.array_equal(lfsr.state, [0, 0, 0, 0, 0, 0, 1])

    lfsr = galois.LFSR(poly, state=4, config="galois")
    assert np.array_equal(lfsr.state, [0, 0, 0, 0, 1, 0, 0])


def test_str():
    poly = galois.Poly.Degrees([7,1,0])
    lfsr = galois.LFSR(poly, config="galois")
    assert str(lfsr) == "<Galois LFSR: poly=Poly(x^7 + x + 1, GF(2))>"


def test_repr():
    poly = galois.Poly.Degrees([7,1,0])
    lfsr = galois.LFSR(poly, config="galois")
    assert repr(lfsr) == "<Galois LFSR: poly=Poly(x^7 + x + 1, GF(2))>"


def test_step_exceptions():
    poly = galois.Poly.Degrees([7,1,0])
    lfsr = galois.LFSR(poly, config="galois")

    with pytest.raises(TypeError):
        lfsr.step(10.0)
    with pytest.raises(ValueError):
        lfsr.step(0)
    with pytest.raises(ValueError):
        lfsr.step(-1)


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


def test_berlekamp_massey_exceptions():
    GF = galois.GF2
    s = GF([0,0,1,1,0,1,1,1,0,1])

    with pytest.raises(TypeError):
        galois.berlekamp_massey(s.view(np.ndarray))
    with pytest.raises(TypeError):
        galois.berlekamp_massey(s, config=1)
    with pytest.raises(TypeError):
        galois.berlekamp_massey(s, state=1)

    with pytest.raises(ValueError):
        galois.berlekamp_massey(np.atleast_2d(s))
    with pytest.raises(ValueError):
        galois.berlekamp_massey(s, config="invalid-argument")


def test_berlekamp_massey_gf2_1():
    """
    Sage:
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
    Sage:
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
    Sage:
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
