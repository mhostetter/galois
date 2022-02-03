"""
A pytest module to test the creation of special polynomials over finite fields.
"""
import pytest

import galois


def test_is_irreducible_exceptions():
    with pytest.raises(TypeError):
        galois.is_irreducible([1, 0, 1, 1])


def test_is_irreducible(poly_is_irreducible):
    GF, IS, IS_NOT = poly_is_irreducible["GF"], poly_is_irreducible["IS"], poly_is_irreducible["IS_NOT"]
    for i in range(len(IS)):
        p = IS[i]
        assert galois.is_irreducible(p)
        p = IS_NOT[i]
        assert not galois.is_irreducible(p)


###############################################################################
# NOTE: Irreducible polynomial tests are in a separate file
###############################################################################


def test_is_primitive_exceptions():
    with pytest.raises(TypeError):
        galois.is_primitive([1, 0, 1, 1])


def test_is_primitive(poly_is_primitive):
    GF, IS, IS_NOT = poly_is_primitive["GF"], poly_is_primitive["IS"], poly_is_primitive["IS_NOT"]

    # FIXME: Skip large fields because they're too slow
    if GF.order > 2**16:
        return

    for i in range(len(IS)):
        p = IS[i]
        assert galois.is_primitive(p)
        p = IS_NOT[i]
        assert not galois.is_primitive(p)
