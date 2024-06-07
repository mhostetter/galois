"""
A pytest module to test the printing options of the package.
"""

import pytest

import galois


def test_default_options():
    assert galois.get_printoptions() == {"coeffs": "desc"}


def test_cant_modify_return_value():
    options = galois.get_printoptions()
    options["new_key"] = 1
    assert "new_key" not in galois.get_printoptions()


def test_set_exceptions():
    with pytest.raises(ValueError):
        galois.set_printoptions(coeffs="invalid")


def test_set_coeffs():
    GF = galois.GF(3**5, repr="poly")
    a = GF(83)
    f = galois.Poly([3, 0, 5, 2], field=galois.GF(7))

    galois.set_printoptions()
    assert galois.get_printoptions()["coeffs"] == "desc"
    assert str(a) == "α^4 + 2"
    assert str(f) == "3x^3 + 5x + 2"

    galois.set_printoptions(coeffs="asc")
    assert galois.get_printoptions()["coeffs"] == "asc"
    assert str(a) == "2 + α^4"
    assert str(f) == "2 + 5x + 3x^3"

    galois.set_printoptions()
    assert galois.get_printoptions()["coeffs"] == "desc"


def test_context_manager():
    GF = galois.GF(3**5, repr="poly")
    a = GF(83)
    f = galois.Poly([3, 0, 5, 2], field=galois.GF(7))
    galois.set_printoptions()  # Ensure the default options are set

    assert galois.get_printoptions()["coeffs"] == "desc"
    assert str(a) == "α^4 + 2"
    assert str(f) == "3x^3 + 5x + 2"

    with galois.printoptions(coeffs="asc"):
        assert galois.get_printoptions()["coeffs"] == "asc"
        assert str(a) == "2 + α^4"
        assert str(f) == "2 + 5x + 3x^3"

    assert galois.get_printoptions()["coeffs"] == "desc"
    assert str(a) == "α^4 + 2"
    assert str(f) == "3x^3 + 5x + 2"
