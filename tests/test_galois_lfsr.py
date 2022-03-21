"""
A pytest module to test the Galois LFSR implementation.
"""
import pytest
import numpy as np

import galois


def test_exceptions():
    c = galois.primitive_poly(7, 4)

    with pytest.raises(TypeError):
        galois.GLFSR(c.reverse().coeffs)
    with pytest.raises(TypeError):
        galois.GLFSR(c.reverse(), state=1)


def test_repr():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())
    assert repr(lfsr) == "<Galois LFSR: f(x) = 5x^4 + 3x^3 + x^2 + 1 over GF(7)>"


def test_str():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())
    assert str(lfsr) == "Galois LFSR:\n  field: GF(7)\n  feedback_poly: 5x^4 + 3x^3 + x^2 + 1\n  characteristic_poly: x^4 + x^2 + 3x + 5\n  taps: [2, 4, 6, 0]\n  order: 4\n  state: [1, 1, 1, 1]\n  initial_state: [1, 1, 1, 1]"


def test_initial_state():
    c = galois.primitive_poly(7, 4)

    default_state = [1, 1, 1, 1]
    lfsr = galois.GLFSR(c.reverse())
    assert np.array_equal(lfsr.initial_state, default_state)
    assert np.array_equal(lfsr.state, default_state)

    state = [1, 2, 3, 4]
    lfsr = galois.GLFSR(c.reverse(), state=state)
    assert np.array_equal(lfsr.initial_state, state)
    assert np.array_equal(lfsr.state, state)


def test_feedback_and_characteristic_poly():
    c = galois.primitive_poly(7, 4)
    f = c.reverse()
    lfsr = galois.GLFSR(f)
    assert lfsr.feedback_poly == f
    assert lfsr.characteristic_poly == c
    assert lfsr.feedback_poly == lfsr.characteristic_poly.reverse()


def test_reset_exceptions():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())

    with pytest.raises(TypeError):
        lfsr.reset(1)


def test_reset_initial_state():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())

    assert np.array_equal(lfsr.state, lfsr.initial_state)
    lfsr.step(10)
    assert not np.array_equal(lfsr.state, lfsr.initial_state)
    lfsr.reset()
    assert np.array_equal(lfsr.state, lfsr.initial_state)


def test_reset_specific_state():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())
    state = [1, 2, 3, 4]

    assert not np.array_equal(lfsr.state, state)
    lfsr.reset(state)
    assert np.array_equal(lfsr.state, state)


def test_step_exceptions():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())

    with pytest.raises(TypeError):
        lfsr.step(10.0)


def test_step_zero():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())

    y = lfsr.step(0)
    assert y.size == 0
    assert type(y) is lfsr.field


def test_step_forwards_backwards():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())

    y_forward = lfsr.step(10)
    y_reverse = lfsr.step(-10)

    assert np.array_equal(y_forward, y_reverse[::-1])
    assert np.array_equal(lfsr.state, lfsr.initial_state)


def test_step_backwards_forwards():
    c = galois.primitive_poly(7, 4)
    lfsr = galois.GLFSR(c.reverse())

    y_reverse = lfsr.step(-10)
    y_forward = lfsr.step(10)

    assert np.array_equal(y_reverse, y_forward[::-1])
    assert np.array_equal(lfsr.state, lfsr.initial_state)


def test_step_gf2_extension_field():
    """
    The states of the degree-n Galois LFSR over GF(q) generate the GF(q^n) extension field with the characteristic polynomial as its
    irreducible polynomial.
    """
    c = galois.conway_poly(2, 4)
    state = [1, 0, 0, 0]
    lfsr = galois.GLFSR(c.reverse(), state=state)

    GF = galois.GF(2**4, irreducible_poly=c)
    alpha = GF.primitive_element

    for i in range(50):
        np.array_equal(lfsr.state[::-1], (alpha**i).vector())
        lfsr.step()


def test_step_gf3_extension_field():
    """
    The states of the degree-n Galois LFSR over GF(q) generate the GF(q^n) extension field with the characteristic polynomial as its
    irreducible polynomial.
    """
    c = galois.conway_poly(3, 4)
    state = [1, 0, 0, 0]
    lfsr = galois.GLFSR(c.reverse(), state=state)

    GF = galois.GF(3**4, irreducible_poly=c)
    alpha = GF.primitive_element

    for i in range(50):
        np.array_equal(lfsr.state[::-1], (alpha**i).vector())
        lfsr.step()


# TODO: Figure out how to compare and/or construct GF((p^m)^n)

# def test_step_gf2_3_extension_field():
#     """
#     The states of the degree-n Galois LFSR over GF(q) generate the GF(q^n) extension field with the characteristic polynomial as its
#     irreducible polynomial.
#     """
#     c = galois.conway_poly(2, 4)
#     c = galois.Poly(c.coeffs, field=galois.GF(2**3))  # Lift c(x) to GF(2^3)
#     state = [1, 0, 0, 0]
#     lfsr = galois.GLFSR(c.reverse(), state=state)

#     GF = galois.GF(2**(3*4), irreducible_poly=galois.conway_poly(2, 3*4))
#     alpha = GF.primitive_element

#     for i in range(50):
#         np.array_equal(lfsr.state[::-1], (alpha**i).vector())
#         lfsr.step()
