"""
A pytest module to test the Berlekamp-Massey algorithm.

Notes
-----
I discovered a bug in Sage's Berlekmap-Massey implementation and filed it here https://trac.sagemath.org/ticket/33537.
"""

import numpy as np
import pytest

import galois


def test_berlekamp_massey_exceptions():
    GF = galois.GF(2)
    s = GF([0, 0, 1, 1, 0, 1, 1, 1, 0, 1])

    with pytest.raises(TypeError):
        galois.berlekamp_massey(s.view(np.ndarray))
    with pytest.raises(TypeError):
        galois.berlekamp_massey(s, output=1)

    with pytest.raises(ValueError):
        galois.berlekamp_massey(np.atleast_2d(s))
    with pytest.raises(ValueError):
        galois.berlekamp_massey(s, output="invalid-argument")


def test_gf2_primitive():
    """
    Python:
        c = galois.primitive_poly(2, 4)
        lfsr = galois.FLFSR(c)
        y = lfsr.step(50)

    Sage:
        F = GF(2)
        y = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
        y = [F(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(2)
    y = GF([0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0])  # fmt: skip
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str("x^4 + x + 1", field=GF)


def test_gf3_primitive():
    """
    Python:
        c = galois.primitive_poly(3, 4)
        lfsr = galois.FLFSR(c)
        y = lfsr.step(50)

    Sage:
        F = GF(3)
        y = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 2, 0, 0, 2, 2, 0, 1, 0, 2, 2, 1, 1, 0, 1, 0, 1, 2, 1, 2, 2, 1, 2, 0, 1, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0]
        y = [F(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(3)
    y = GF([1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 2, 0, 0, 2, 2, 0, 1, 0, 2, 2, 1, 1, 0, 1, 0, 1, 2, 1, 2, 2, 1, 2, 0, 1, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0])  # fmt: skip
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str("x^4 + x + 2", field=GF)


def test_gf2_3_primitive():
    """
    Python:
        c = galois.primitive_poly(2**3, 4)
        lfsr = galois.FLFSR(c)
        y = lfsr.step(50)

    Sage:
        F = GF(2^3, repr="int")
        y = [1, 1, 1, 1, 2, 2, 2, 1, 4, 4, 7, 7, 3, 0, 5, 1, 5, 5, 5, 6, 1, 1, 2, 0, 2, 1, 6, 2, 7, 5, 3, 1, 7, 7, 4, 4, 5, 6, 3, 2, 2, 2, 7, 4, 4, 1, 6, 3, 6, 5]
        y = [F.fetch_int(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(2**3)
    y = GF([1, 1, 1, 1, 2, 2, 2, 1, 4, 4, 7, 7, 3, 0, 5, 1, 5, 5, 5, 6, 1, 1, 2, 0, 2, 1, 6, 2, 7, 5, 3, 1, 7, 7, 4, 4, 5, 6, 3, 2, 2, 2, 7, 4, 4, 1, 6, 3, 6, 5])  # fmt: skip
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str("x^4 + x + 3", field=GF)


def test_gf3_3_primitive():
    """
    Python:
        c = galois.primitive_poly(3**3, 4)
        lfsr = galois.FLFSR(c)
        y = lfsr.step(50)

    Sage:
        F = GF(3^3, repr="int")
        y = [1, 1, 1, 1, 19, 19, 19, 1, 25, 25, 16, 4, 24, 6, 6, 6, 26, 2, 2, 9, 4, 11, 1, 11, 13, 21, 9, 9, 12, 10, 3, 0, 6, 2, 4, 3, 6, 15, 18, 7, 20, 20, 20, 8, 17, 17, 2, 1, 13, 19]
        y = [F.fetch_int(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(3**3)
    y = GF([1, 1, 1, 1, 19, 19, 19, 1, 25, 25, 16, 4, 24, 6, 6, 6, 26, 2, 2, 9, 4, 11, 1, 11, 13, 21, 9, 9, 12, 10, 3, 0, 6, 2, 4, 3, 6, 15, 18, 7, 20, 20, 20, 8, 17, 17, 2, 1, 13, 19])  # fmt: skip
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str("x^4 + x + 10", field=GF)


def test_gf2_random():
    """
    Python:
        GF = galois.GF(2)
        y = GF.Random(50)

    Sage:
        F = GF(2)
        y = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
        y = [F(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(2)
    y = GF([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1])  # fmt: skip
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str(
        "x^24 + x^21 + x^19 + x^18 + x^17 + x^15 + x^14 + x^13 + x^12 + x^11 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + x + 1",
        field=GF,
    )


def test_gf3_random():
    """
    Python:
        GF = galois.GF(3)
        y = GF.Random(50)

    Sage:
        F = GF(3)
        y = [1, 0, 1, 2, 0, 0, 0, 0, 2, 0, 1, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0, 1, 0, 2, 2, 0, 2, 0, 1, 0, 2, 2, 2, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 2, 2, 1, 2, 1]
        y = [F(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(3)
    y = GF([1, 0, 1, 2, 0, 0, 0, 0, 2, 0, 1, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0, 1, 0, 2, 2, 0, 2, 0, 1, 0, 2, 2, 2, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 2, 2, 1, 2, 1])  # fmt: skip
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str(
        "x^25 + 2*x^24 + x^22 + x^20 + 2*x^18 + x^17 + 2*x^14 + 2*x^13 + x^12 + 2*x^11 + 2*x^10 + x^9 + 2*x^8 + x^7 + 2*x^6 + 2*x^2 + 2*x + 2",
        field=GF,
    )


def test_gf2_3_random():
    """
    Random sequences with characteristic polynomial with degree 5

    Python:
        GF = galois.GF(2**3)
        y = GF.Random(10)
        print(y.tolist())

    Sage:
        F = GF(2^3, repr="int")
        y = [F.fetch_int(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(2**3)

    y = GF([1, 1, 3, 0, 4, 1, 5, 3, 1, 5])
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str("x^5 + 4*x^4 + 2*x^3 + 5*x^2 + 5*x + 6", field=GF)

    y = GF([4, 3, 5, 5, 6, 5, 0, 1, 3, 1])
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str("x^5 + 7*x^4 + 2*x^3 + 2*x^2 + 7*x + 2", field=GF)


# def test_gf2_3_random_other():
#     """
#     Random sequences with characteristic polynomial with degree less than 5

#     Sage:
#         F = GF(2^3, repr="int")
#         y = [randint(0, 8) for _ in range(10)]
#         y = [F.fetch_int(k) for k in y]
#         c = berlekamp_massey(y)
#         print(y)
#         print(c)
#     """
#     GF = galois.GF(2**3)

#     y = GF([0, 5, 0, 3, 7, 0, 4, 6, 7, 6])
#     c = galois.berlekamp_massey(y)
#     assert c == galois.Poly.Str("x^2 + 4*x + 6", field=GF)

#     y = GF([4, 0, 2, 0, 7, 2, 0, 0, 2, 1])
#     c = galois.berlekamp_massey(y)
#     assert c == galois.Poly.Str("x^4 + 3*x^3 + 3", field=GF)

#     y = GF([7, 2, 6, 0, 5, 4, 2, 1, 2, 7])
#     c = galois.berlekamp_massey(y)
#     assert c == galois.Poly.Str("x^4 + 6*x^3 + 6*x^2 + 6*x + 7", field=GF)


def test_gf3_3_random():
    """
    Python:
        GF = galois.GF(3**3)
        y = GF.Random(50)

    Sage:
        F = GF(3^3, repr="int")
        y = [1, 12, 12, 3, 23, 3, 17, 4, 16, 7, 19, 10, 16, 1, 25, 12, 13, 6, 1, 26, 17, 7, 15, 26, 10, 19, 22, 11, 19, 18, 18, 23, 22, 24, 13, 0, 11, 6, 11, 20, 23, 13, 5, 22, 8, 25, 0, 10, 7, 1]
        y = [F.fetch_int(yi) for yi in y]
        berlekamp_massey(y)
    """
    GF = galois.GF(3**3)
    y = GF([1, 12, 12, 3, 23, 3, 17, 4, 16, 7, 19, 10, 16, 1, 25, 12, 13, 6, 1, 26, 17, 7, 15, 26, 10, 19, 22, 11, 19, 18, 18, 23, 22, 24, 13, 0, 11, 6, 11, 20, 23, 13, 5, 22, 8, 25, 0, 10, 7, 1])  # fmt: skip
    c = galois.berlekamp_massey(y)
    assert c == galois.Poly.Str(
        "x^25 + 17*x^24 + 26*x^23 + 25*x^22 + 2*x^21 + 5*x^20 + 17*x^19 + 25*x^18 + 12*x^17 + 16*x^16 + 6*x^15 + 10*x^14 + 23*x^13 + x^12 + 3*x^11 + 12*x^10 + 4*x^9 + 3*x^8 + 24*x^7 + 26*x^6 + x^5 + 2*x^4 + 24*x^3 + 5*x^2 + 22*x + 24",
        field=GF,
    )


@pytest.mark.parametrize("order", [2, 3, 2**3, 3**3])
def test_fibonacci_lfsr_primitive(order):
    GF = galois.GF(order)
    n = 4
    c = galois.primitive_poly(order, n, method="random")
    state = GF.Random(4)
    state[0] = GF.Random(low=1)  # Ensure state is non-zero
    lfsr = galois.FLFSR(c.reverse(), state=state)
    y = lfsr.step(2 * n)

    lfsr_2 = galois.berlekamp_massey(y, output="fibonacci")
    assert type(lfsr_2) is galois.FLFSR
    assert lfsr_2.feedback_poly == lfsr.feedback_poly
    assert lfsr_2.characteristic_poly == lfsr.characteristic_poly
    assert np.array_equal(lfsr_2.initial_state, lfsr.initial_state)


@pytest.mark.parametrize("order", [2, 3, 2**3, 3**3])
def test_galois_lfsr_primitive(order):
    GF = galois.GF(order)
    n = 4
    c = galois.primitive_poly(order, n, method="random")
    state = GF.Random(4)
    state[0] = GF.Random(low=1)  # Ensure state is non-zero
    lfsr = galois.GLFSR(c.reverse(), state=state)
    y = lfsr.step(2 * n)

    lfsr_2 = galois.berlekamp_massey(y, output="galois")
    assert type(lfsr_2) is galois.GLFSR
    assert lfsr_2.feedback_poly == lfsr.feedback_poly
    assert lfsr_2.characteristic_poly == lfsr.characteristic_poly
    assert np.array_equal(lfsr_2.initial_state, lfsr.initial_state)


# @pytest.mark.parametrize("order", [2, 3, 2**3, 3**3])
# def test_fibonacci_lfsr_random(order):
#     GF = galois.GF(order)
#     y = GF.Random(20)

#     lfsr = galois.berlekamp_massey(y, output="fibonacci")
#     assert type(lfsr) is galois.FLFSR
#     z = lfsr.step(20)
#     assert np.array_equal(z, y)


# @pytest.mark.parametrize("order", [2, 3, 2**3, 3**3])
# def test_galois_lfsr_random(order):
#     GF = galois.GF(order)
#     y = GF.Random(20)

#     lfsr = galois.berlekamp_massey(y, output="galois")
#     assert type(lfsr) is galois.GLFSR
#     z = lfsr.step(20)
#     assert np.array_equal(z, y)
