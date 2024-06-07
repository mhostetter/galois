"""
A pytest module to test the class factory function :obj:`galois.GF`.
"""

import numpy as np
import pytest

import galois


@pytest.mark.parametrize("characteristic,degree", [(2, 1), (2, 8), (3, 1), (3, 5)])
def test_defaults(characteristic, degree):
    GF = galois.GF(characteristic**degree)
    assert issubclass(GF, galois.FieldArray)
    assert GF.characteristic == characteristic
    assert GF.degree == degree

    GF = galois.GF(characteristic, degree)
    assert issubclass(GF, galois.FieldArray)
    assert GF.characteristic == characteristic
    assert GF.degree == degree


def test_mandatory_kwargs():
    GF = galois.GF(3**5, irreducible_poly="x^5 + 2x + 1")
    assert GF.order == 3**5
    assert GF.irreducible_poly == "x^5 + 2x + 1"

    with pytest.raises(TypeError):
        galois.GF(3**5, "x^5 + 2x + 1")

    GF = galois.GF(3, 5, irreducible_poly="x^5 + 2x + 1")
    assert GF.order == 3**5
    assert GF.irreducible_poly == "x^5 + 2x + 1"

    with pytest.raises(TypeError):
        galois.GF(3, 5, "x^5 + 2x + 1")


def test_defaults_dont_modify_ufunc_mode():
    """
    Ensures ufunc_mode=None (the default) doesn't modify the current ufunc mode.
    """
    GF = galois.GF(2**8)
    GF.compile("auto")  # Reset to default
    assert GF.ufunc_mode == "jit-lookup"
    GF.compile("jit-calculate")
    assert GF.ufunc_mode == "jit-calculate"
    GF = galois.GF(2**8)
    assert GF.ufunc_mode == "jit-calculate"
    GF.compile("auto")  # Reset to default
    assert GF.ufunc_mode == "jit-lookup"


def test_can_modify_ufunc_mode():
    GF = galois.GF(2**8)
    GF.compile("auto")  # Reset to default
    assert GF.ufunc_mode == "jit-lookup"
    GF = galois.GF(2**8, compile="jit-calculate")
    assert GF.ufunc_mode == "jit-calculate"
    GF.compile("auto")  # Reset to default
    assert GF.ufunc_mode == "jit-lookup"


def test_defaults_dont_modify_element_repr():
    """
    Ensures repr=None (the default) doesn't modify the current element representation.
    """
    GF = galois.GF(2**8)
    GF.repr()  # Reset to default
    assert GF.element_repr == "int"
    GF.repr("poly")
    assert GF.element_repr == "poly"
    GF = galois.GF(2**8)
    assert GF.element_repr == "poly"
    GF.repr()  # Reset to default
    assert GF.element_repr == "int"


def test_can_modify_element_repr():
    GF = galois.GF(2**8)
    GF.repr()  # Reset to default
    assert GF.element_repr == "int"
    GF = galois.GF(2**8, repr="poly")
    assert GF.element_repr == "poly"
    GF.repr()  # Reset to default
    assert GF.element_repr == "int"


def test_basic_exceptions():
    with pytest.raises(TypeError):
        galois.GF(2.0**8)
    with pytest.raises(TypeError):
        galois.GF(2**8, verify=1)
    with pytest.raises(TypeError):
        galois.GF(2**8, compile=True)
    with pytest.raises(TypeError):
        galois.GF(2**8, repr=True)

    with pytest.raises(ValueError):
        galois.GF(2**3 * 5**3)
    with pytest.raises(ValueError):
        galois.GF(2**8, compile="invalid-argument")
    with pytest.raises(ValueError):
        galois.GF(2**8, repr="invalid-argument")


def test_irreducible_poly_exceptions():
    with pytest.raises(TypeError):
        galois.GF(2**8, irreducible_poly=285.0)
    with pytest.raises(ValueError):
        galois.GF(2**8, irreducible_poly=galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1], field=galois.GF(3)))
    with pytest.raises(ValueError):
        galois.GF(2**8, irreducible_poly=galois.Poly([1, 1, 0, 0, 0, 1, 1, 1, 0, 1]))
    with pytest.raises(ValueError):
        galois.GF(2**8, irreducible_poly=galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 0]))

    # Don't allow `irreducible_poly` for prime fields
    with pytest.raises(ValueError):
        galois.GF(3, irreducible_poly=[1, 1])


def test_primitive_element_exceptions():
    with pytest.raises(TypeError):
        galois.GF(2**8, primitive_element=2.0)
    with pytest.raises(ValueError):
        galois.GF(2**8, primitive_element=256)
    with pytest.raises(ValueError):
        galois.GF(7, primitive_element=10)
    with pytest.raises(ValueError):
        galois.GF(7, primitive_element=4)
    with pytest.raises(ValueError):
        galois.GF(2**8, primitive_element=galois.Poly([1, 0], field=galois.GF(3)))
    with pytest.raises(ValueError):
        galois.GF(2**8, primitive_element=galois.Poly([1, 0], field=galois.GF(3)))
    with pytest.raises(ValueError):
        galois.GF(2**8, primitive_element=galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]))
    with pytest.raises(ValueError):
        galois.GF(2**8, primitive_element=galois.Poly([1, 1, 1]))


@pytest.mark.parametrize("characteristic,degree", [(2, 8), (3, 5)])
def test_specify_irreducible_poly(characteristic, degree):
    GF = galois.GF(characteristic**degree)
    poly = GF.irreducible_poly
    assert galois.GF(characteristic**degree, irreducible_poly=int(poly)) is GF
    assert galois.GF(characteristic**degree, irreducible_poly=str(poly)) is GF
    assert galois.GF(characteristic**degree, irreducible_poly=tuple(poly.coeffs)) is GF
    assert galois.GF(characteristic**degree, irreducible_poly=list(poly.coeffs)) is GF
    assert galois.GF(characteristic**degree, irreducible_poly=np.array(poly.coeffs)) is GF
    assert galois.GF(characteristic**degree, irreducible_poly=poly.coeffs) is GF
    assert galois.GF(characteristic**degree, irreducible_poly=poly) is GF


@pytest.mark.parametrize("characteristic,degree", [(2, 1), (3, 1)])
def test_specify_primitive_element_prime(characteristic, degree):
    GF = galois.GF(characteristic**degree)
    alpha = GF.primitive_element
    assert galois.GF(characteristic**degree, primitive_element=int(alpha)) is GF


@pytest.mark.parametrize("characteristic,degree", [(2, 8), (3, 5)])
def test_specify_primitive_element_extension(characteristic, degree):
    GF = galois.GF(characteristic**degree)
    poly = galois.Poly(GF.primitive_element.vector())
    assert galois.GF(characteristic**degree, primitive_element=int(poly)) is GF
    assert galois.GF(characteristic**degree, primitive_element=str(poly)) is GF
    assert galois.GF(characteristic**degree, primitive_element=tuple(poly.coeffs)) is GF
    assert galois.GF(characteristic**degree, primitive_element=list(poly.coeffs)) is GF
    assert galois.GF(characteristic**degree, primitive_element=np.array(poly.coeffs)) is GF
    assert galois.GF(characteristic**degree, primitive_element=poly.coeffs) is GF
    assert galois.GF(characteristic**degree, primitive_element=poly) is GF
