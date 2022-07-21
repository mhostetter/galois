"""
A pytest module to test Reed-Solomon codes.

Test vectors generated from Octave with rsgenpoly().

References
----------
* https://octave.sourceforge.io/communications/function/rsgenpoly.html
"""
import pytest
import numpy as np

import galois


def test_rs_exceptions():
    with pytest.raises(TypeError):
        galois.ReedSolomon(15.0, 11)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 11.0)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 11, c=1.0)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 11, primitive_poly=19.0)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 11, systematic=1)

    with pytest.raises(ValueError):
        galois.ReedSolomon(15, 12)
    with pytest.raises(ValueError):
        galois.ReedSolomon(14, 12)
    with pytest.raises(ValueError):
        galois.ReedSolomon(15, 11, c=0)


def test_repr():
    rs = galois.ReedSolomon(15, 11)
    assert repr(rs) == "<Reed-Solomon Code: [15, 11, 5] over GF(2^4)>"


def test_str():
    rs = galois.ReedSolomon(15, 11)
    assert str(rs) == "Reed-Solomon Code:\n  [n, k, d]: [15, 11, 5]\n  field: GF(2^4)\n  generator_poly: x^4 + 13x^3 + 12x^2 + 8x + 7\n  is_narrow_sense: True\n  is_systematic: True\n  t: 2"


def test_rs_generator_poly():
    # S. Lin and D. Costello. Error Control Coding. Example 7.1, p. 238.
    p = galois.primitive_poly(2, 6)
    GF = galois.GF(2**6, irreducible_poly=p)
    a = GF.primitive_element
    assert galois.ReedSolomon(63, 57).generator_poly == galois.Poly(a**np.array([0, 59, 48, 43, 55, 10, 21]), field=GF)

    # Octave rsgenpoly()
    assert np.array_equal(galois.ReedSolomon(15, 13).generator_poly.coeffs, [1, 6, 8])
    assert np.array_equal(galois.ReedSolomon(15, 11).generator_poly.coeffs, [1, 13, 12, 8, 7])
    assert np.array_equal(galois.ReedSolomon(15, 9).generator_poly.coeffs, [1, 7, 9, 3, 12, 10, 12])
    assert np.array_equal(galois.ReedSolomon(15, 7).generator_poly.coeffs, [1, 9, 4, 3, 4, 13, 6, 14, 12])
    assert np.array_equal(galois.ReedSolomon(15, 5).generator_poly.coeffs, [1, 4, 8, 10, 12, 9, 4, 2, 12, 2, 7])
    assert np.array_equal(galois.ReedSolomon(15, 3).generator_poly.coeffs, [1, 5, 9, 5, 8, 1, 4, 13, 9, 4, 12, 13, 8])
    assert np.array_equal(galois.ReedSolomon(15, 1).generator_poly.coeffs, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Ensure we're using the correct default primitive polynomial
    assert np.array_equal(galois.ReedSolomon(7, 5).generator_poly.coeffs, [1, 6, 3])
    assert np.array_equal(galois.ReedSolomon(15, 9).generator_poly.coeffs, [1, 7, 9, 3, 12, 10, 12])
    assert np.array_equal(galois.ReedSolomon(31, 25).generator_poly.coeffs, [1, 17, 26, 30, 27, 30, 24])
    assert np.array_equal(galois.ReedSolomon(63, 57).generator_poly.coeffs, [1, 61, 13, 55, 46, 48, 59])
    assert np.array_equal(galois.ReedSolomon(127, 121).generator_poly.coeffs, [1, 126,  64,  68, 100,  34, 109])
    assert np.array_equal(galois.ReedSolomon(255, 249).generator_poly.coeffs, [1, 126,   4, 158,  58,  49, 117])
    assert np.array_equal(galois.ReedSolomon(511, 505).generator_poly.coeffs, [1, 126, 254, 108, 222,  26,  76])
    # NOTE: Disabling because creation of large generator matrices currently takes a long time
    # assert np.array_equal(galois.ReedSolomon(1023, 1017).generator_poly.coeffs, [1, 126, 131, 847, 272, 158, 130])
    # assert np.array_equal(galois.ReedSolomon(2047, 2041).generator_poly.coeffs, [1, 126, 1181, 1719, 2029, 1077, 1034])
    # assert np.array_equal(galois.ReedSolomon(8191, 8185).generator_poly.coeffs, [1, 126, 3224, 7834, 3814, 2340, 6912])
    # assert np.array_equal(galois.ReedSolomon(2**14 - 1, 2**14 - 7).generator_poly.coeffs, [1, 126, 3224, 4613, 10792, 15051, 920])
    # assert np.array_equal(galois.ReedSolomon(2**15 - 1, 2**15 - 7).generator_poly.coeffs, [1, 126, 3224, 24259, 19476, 65, 192])
    # assert np.array_equal(galois.ReedSolomon(2**16 - 1, 2**16 - 7).generator_poly.coeffs, [1, 126, 3224, 57024, 11322, 24786, 8566])


def test_rs_generator_poly_specify_c():
    # Octave rsgenpoly(15, k, 19, 2)
    assert np.array_equal(galois.ReedSolomon(15, 13, c=2).generator_poly.coeffs, [1, 12, 6])
    assert np.array_equal(galois.ReedSolomon(15, 11, c=2).generator_poly.coeffs, [1, 9, 5, 12, 9])

    # Octave rsgenpoly(15, k, 19, 3)
    assert np.array_equal(galois.ReedSolomon(15, 13, c=3).generator_poly.coeffs, [1, 11, 11])
    assert np.array_equal(galois.ReedSolomon(15, 11, c=3).generator_poly.coeffs, [1, 1, 7, 10, 8])

    # Octave rsgenpoly(15, k, 19, 4)
    assert np.array_equal(galois.ReedSolomon(15, 13, c=4).generator_poly.coeffs, [1, 5, 10])
    assert np.array_equal(galois.ReedSolomon(15, 11, c=4).generator_poly.coeffs, [1, 2, 15, 15, 11])


def test_rs_generator_poly_specify_primitive_poly():
    # Octave rsgenpoly(31, k, 41)
    p = galois.Poly.Degrees([5,3,0])
    assert np.array_equal(galois.ReedSolomon(31, 29, primitive_poly=p).generator_poly.coeffs, [1, 6, 8])
    assert np.array_equal(galois.ReedSolomon(31, 27, primitive_poly=p).generator_poly.coeffs, [1, 30, 7, 24, 19])

    # Octave rsgenpoly(31, k, 41, 2)
    p = galois.Poly.Degrees([5,3,0])
    assert np.array_equal(galois.ReedSolomon(31, 29, primitive_poly=p, c=2).generator_poly.coeffs, [1, 12, 9])
    assert np.array_equal(galois.ReedSolomon(31, 27, primitive_poly=p, c=2).generator_poly.coeffs, [1, 21, 28, 31, 3])

    # Octave rsgenpoly(31, k, 61)
    p = galois.Poly.Degrees([5,4,3,2,0])
    assert np.array_equal(galois.ReedSolomon(31, 29, primitive_poly=p).generator_poly.coeffs, [1, 6, 8])
    assert np.array_equal(galois.ReedSolomon(31, 27, primitive_poly=p).generator_poly.coeffs, [1, 30, 17, 16, 10])

    # Octave rsgenpoly(31, k, 61, 2)
    p = galois.Poly.Degrees([5,4,3,2,0])
    assert np.array_equal(galois.ReedSolomon(31, 29, primitive_poly=p, c=2).generator_poly.coeffs, [1, 12, 29])
    assert np.array_equal(galois.ReedSolomon(31, 27, primitive_poly=p, c=2).generator_poly.coeffs, [1, 1, 3, 14, 19])


def test_rs_parity_check_matrix():
    p = galois.Poly.Degrees([4,1,0])
    GF = galois.GF(2**4, irreducible_poly=p)
    alpha = GF.primitive_element
    rs = galois.ReedSolomon(15, 11)
    H_truth = alpha**np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
    ])
    assert np.array_equal(rs.H, np.fliplr(H_truth))  # NOTE: We use the convention of polynomial highest degree first, not last


def test_rs_properties():
    rs = galois.ReedSolomon(7, 5)
    assert (rs.n, rs.k, rs.t) == (7, 5, 1)
    assert rs.c == 1
    assert rs.is_narrow_sense == True

    rs = galois.ReedSolomon(15, 11)
    assert (rs.n, rs.k, rs.t) == (15, 11, 2)
    assert rs.c == 1
    assert rs.is_narrow_sense == True

    rs = galois.ReedSolomon(15, 7)
    assert (rs.n, rs.k, rs.t) == (15, 7, 4)
    assert rs.c == 1
    assert rs.is_narrow_sense == True

    rs = galois.ReedSolomon(15, 5)
    assert (rs.n, rs.k, rs.t) == (15, 5, 5)
    assert rs.c == 1
    assert rs.is_narrow_sense == True
