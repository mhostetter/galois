"""
A pytest module to test methods of Galois field array classes.
"""
import pytest

import galois


def test_display_method():
    GF = galois.GF(2**3)
    a = GF([1, 0, 5, 2])
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"
    GF.display("poly")
    assert str(a) == "GF([1, 0, α^2 + 1, α], order=2^3)"
    GF.display("power")
    assert str(a) == "GF([1, 0, α^6, α], order=2^3)"
    GF.display()
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"


def test_display_context_manager():
    GF = galois.GF(2**3)
    a = GF([1, 0, 5, 2])
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"
    with GF.display("poly"):
        assert str(a) == "GF([1, 0, α^2 + 1, α], order=2^3)"
    with GF.display("power"):
        assert str(a) == "GF([1, 0, α^6, α], order=2^3)"
    assert str(a) == "GF([1, 0, 5, 2], order=2^3)"


def test_display_exceptions():
    GF = galois.GF(2**3)
    a = GF([1, 0, 5, 2])
    with pytest.raises(ValueError):
        GF.display("invalid-display-type")


def test_arithmetic_table():
    GF = galois.GF2
    assert GF.arithmetic_table("+") == "╔═══════╦═══╦═══╗\n║ x + y ║ 0 │ 1 ║\n╠═══════╬═══╬═══╣\n║     0 ║ 0 │ 1 ║\n╟───────╫───┼───╢\n║     1 ║ 1 │ 0 ║\n╚═══════╩═══╩═══╝"
    assert GF.arithmetic_table("-") == "╔═══════╦═══╦═══╗\n║ x - y ║ 0 │ 1 ║\n╠═══════╬═══╬═══╣\n║     0 ║ 0 │ 1 ║\n╟───────╫───┼───╢\n║     1 ║ 1 │ 0 ║\n╚═══════╩═══╩═══╝"
    assert GF.arithmetic_table("*") == "╔═══════╦═══╦═══╗\n║ x * y ║ 0 │ 1 ║\n╠═══════╬═══╬═══╣\n║     0 ║ 0 │ 0 ║\n╟───────╫───┼───╢\n║     1 ║ 0 │ 1 ║\n╚═══════╩═══╩═══╝"
    assert GF.arithmetic_table("/") == "╔═══════╦═══╗\n║ x / y ║ 1 ║\n╠═══════╬═══╣\n║     0 ║ 0 ║\n╟───────╫───╢\n║     1 ║ 1 ║\n╚═══════╩═══╝"


def test_repr_table():
    GF = galois.GF2
    assert GF.repr_table() == "╔═══════╦════════════╦════════╦═════════╗\n║ Power │ Polynomial │ Vector │ Integer ║\n║═══════╬════════════╬════════╬═════════║\n║   0   │     0      │  [0]   │    0    ║\n╟───────┼────────────┼────────┼─────────╢\n║  1^0  │     1      │  [1]   │    1    ║\n╚═══════╩════════════╩════════╩═════════╝"
