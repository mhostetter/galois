"""
A pytest module to test polynomial test (is_blank) method and properties.
"""


def test_is_monic(poly_is_monic):
    X, Z = poly_is_monic["X"], poly_is_monic["Z"]
    for x, z in zip(X, Z):
        assert x.is_monic == z


def test_is_irreducible(poly_is_irreducible):
    IS, IS_NOT = poly_is_irreducible["IS"], poly_is_irreducible["IS_NOT"]
    for p_is, p_is_not in zip(IS, IS_NOT):
        assert p_is.is_irreducible()
        assert not p_is_not.is_irreducible()


def test_is_primitive(poly_is_primitive):
    IS, IS_NOT = poly_is_primitive["IS"], poly_is_primitive["IS_NOT"]
    for p_is, p_is_not in zip(IS, IS_NOT):
        assert p_is.is_primitive()
        assert not p_is_not.is_primitive()


def test_is_square_free(poly_is_square_free):
    X, Z = poly_is_square_free["X"], poly_is_square_free["Z"]
    for x, z in zip(X, Z):
        assert x.is_square_free() == z
