"""
A pytest module to test BCH codes.

Test vectors generated from Octave with bchpoly().

References
----------
* Lin, S. and Costello, D. Error Control Coding. Appendix C, pp. 1231.
* https://link.springer.com/content/pdf/bbm%3A978-1-4899-2174-1%2F1.pdf
* https://octave.sourceforge.io/communications/function/bchpoly.html
"""
import pytest
import numpy as np

import galois


def test_bch_exceptions():
    with pytest.raises(TypeError):
        galois.BCH(15.0, 7)
    with pytest.raises(TypeError):
        galois.BCH(15, 7.0)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, c=1.0)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, primitive_poly=19.0)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, primitive_element=2.0)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, systematic=1)

    with pytest.raises(ValueError):
        galois.BCH(15, 12)
    with pytest.raises(ValueError):
        galois.BCH(14, 7)
    # with pytest.raises(ValueError):
    #     galois.BCH(15, 7, c=0)


def test_repr():
    bch = galois.BCH(15, 7)
    assert repr(bch) == "<BCH Code: [15, 7, 5] over GF(2)>"


def test_str():
    bch = galois.BCH(15, 7)
    assert str(bch) == "BCH Code:\n  [n, k, d]: [15, 7, 5]\n  field: GF(2)\n  generator_poly: x^8 + x^7 + x^6 + x^4 + 1\n  is_primitive: True\n  is_narrow_sense: True\n  is_systematic: True\n  t: 2"


def test_bch_generator_poly_7():
    assert int(galois.BCH(7, 4).generator_poly) == 0o13
    assert int(galois.BCH(7, 1).generator_poly) == 0o177


def test_bch_generator_poly_15():
    assert int(galois.BCH(15, 11).generator_poly) == 0o23
    assert int(galois.BCH(15, 7).generator_poly) == 0o721
    assert int(galois.BCH(15, 5).generator_poly) == 0o2467
    assert int(galois.BCH(15, 1).generator_poly) == 0o77777


def test_bch_generator_poly_31():
    assert int(galois.BCH(31, 26).generator_poly) == 0o45
    assert int(galois.BCH(31, 21).generator_poly) == 0o3551
    assert int(galois.BCH(31, 16).generator_poly) == 0o107657
    assert int(galois.BCH(31, 11).generator_poly) == 0o5423325
    assert int(galois.BCH(31, 1).generator_poly) == 0o17777777777


def test_bch_generator_poly_63():
    assert int(galois.BCH(63, 57).generator_poly) == 0o103
    assert int(galois.BCH(63, 51).generator_poly) == 0o12471
    assert int(galois.BCH(63, 45).generator_poly) == 0o1701317
    assert int(galois.BCH(63, 39).generator_poly) == 0o166623567
    assert int(galois.BCH(63, 36).generator_poly) == 0o1033500423
    assert int(galois.BCH(63, 30).generator_poly) == 0o157464165547
    assert int(galois.BCH(63, 24).generator_poly) == 0o17323260404441
    assert int(galois.BCH(63, 18).generator_poly) == 0o1363026512351725
    assert int(galois.BCH(63, 16).generator_poly) == 0o6331141367235453
    assert int(galois.BCH(63, 10).generator_poly) == 0o472622305527250155
    assert int(galois.BCH(63, 7).generator_poly) == 0o5231045543503271737
    assert int(galois.BCH(63, 1).generator_poly) == 0o777777777777777777777


def test_bch_generator_poly_127():
    assert int(galois.BCH(127, 120).generator_poly) == 0o211
    assert int(galois.BCH(127, 113).generator_poly) == 0o41567
    assert int(galois.BCH(127, 106).generator_poly) == 0o11554_743
    # ...
    assert int(galois.BCH(127, 57).generator_poly) == 0o33526_52525_05705_05351_7721
    assert int(galois.BCH(127, 50).generator_poly) == 0o54446_51252_33140_12421_50142_1
    assert int(galois.BCH(127, 43).generator_poly) == 0o17721_77221_36512_27521_22057_4343
    # ...
    assert int(galois.BCH(127, 15).generator_poly) == 0o22057_04244_56045_54770_52301_37622_17604_353
    assert int(galois.BCH(127, 8).generator_poly) == 0o70472_64052_75103_06514_76224_27156_77331_30217
    assert int(galois.BCH(127, 1).generator_poly) == 0o17777_77777_77777_77777_77777_77777_77777_77777_777


def test_bch_generator_poly_255():
    assert int(galois.BCH(255, 247).generator_poly) == 0o435
    assert int(galois.BCH(255, 239).generator_poly) == 0o26754_3
    assert int(galois.BCH(255, 231).generator_poly) == 0o15672_0665
    # ...
    assert int(galois.BCH(255, 91).generator_poly) == 0o67502_65030_32744_41727_23631_72473_25110_75550_76272_07243_44561
    assert int(galois.BCH(255, 87).generator_poly) == 0o11013_67634_14743_23643_52316_34307_17204_62067_22545_27331_17213_17
    assert int(galois.BCH(255, 79).generator_poly) == 0o66700_03563_76575_00020_27034_42073_66174_62101_53267_11766_54134_2355
    # ...
    assert int(galois.BCH(255, 13).generator_poly) == 0o46417_32005_05256_45444_26573_71425_00660_04330_67744_54765_61403_17467_72135_70261_34460_50054_7
    assert int(galois.BCH(255, 9).generator_poly) == 0o15726_02521_74724_63201_03104_32553_55134_61416_23672_12044_07454_51127_66115_54770_55616_77516_057
    assert int(galois.BCH(255, 1).generator_poly) == 0o77777_77777_77777_77777_77777_77777_77777_77777_77777_77777_77777_77777_77777_77777_77777_77777_77777


def test_bch_generator_poly_1024():
    assert int(galois.BCH(1023, 1013).generator_poly) == 0o2011
    assert int(galois.BCH(1023, 1003).generator_poly) == 0o4014167
    # ...
    # asserint(t galois.BCH(1023, 11).generator_poly) == 0o3435423242053413257500125205705563224605


def test_bch_properties():
    bch = galois.BCH(7, 4)
    assert (bch.n, bch.k, bch.t) == (7, 4, 1)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)

    bch = galois.BCH(15, 11)
    assert (bch.n, bch.k, bch.t) == (15, 11, 1)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(15, 7)
    assert (bch.n, bch.k, bch.t) == (15, 7, 2)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(15, 5)
    assert (bch.n, bch.k, bch.t) == (15, 5, 3)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)

    bch = galois.BCH(31, 26)
    assert (bch.n, bch.k, bch.t) == (31, 26, 1)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(31, 21)
    assert (bch.n, bch.k, bch.t) == (31, 21, 2)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(31, 16)
    assert (bch.n, bch.k, bch.t) == (31, 16, 3)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(31, 11)
    assert (bch.n, bch.k, bch.t) == (31, 11, 5)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(31, 6)
    assert (bch.n, bch.k, bch.t) == (31, 6, 7)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)

    bch = galois.BCH(63, 57)
    assert (bch.n, bch.k, bch.t) == (63, 57, 1)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 51)
    assert (bch.n, bch.k, bch.t) == (63, 51, 2)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 45)
    assert (bch.n, bch.k, bch.t) == (63, 45, 3)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 39)
    assert (bch.n, bch.k, bch.t) == (63, 39, 4)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 36)
    assert (bch.n, bch.k, bch.t) == (63, 36, 5)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 30)
    assert (bch.n, bch.k, bch.t) == (63, 30, 6)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 24)
    assert (bch.n, bch.k, bch.t) == (63, 24, 7)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 18)
    assert (bch.n, bch.k, bch.t) == (63, 18,10)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 16)
    assert (bch.n, bch.k, bch.t) == (63, 16,11)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 10)
    assert (bch.n, bch.k, bch.t) == (63, 10,13)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)
    bch = galois.BCH(63, 7)
    assert (bch.n, bch.k, bch.t) == (63, 7,15)
    assert (bch.is_primitive, bch.is_narrow_sense) == (True, True)


def test_bch_generator_poly_diff_primitive_poly():
    """
    Test with primitive polynomial others than the default. Generated in Octave with `bchpoly()`.
    """
    p = galois.Poly.Degrees([3, 2, 0])  # galois.primitive_poly(2, 3, method="max")
    assert galois.BCH(7, 4, primitive_poly=p).generator_poly == galois.Poly([1,0,1,1], order="asc")

    p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="max")
    assert galois.BCH(15, 11, primitive_poly=p).generator_poly == galois.Poly([1,0,0,1,1], order="asc")
    assert galois.BCH(15, 7, primitive_poly=p).generator_poly == galois.Poly([1,1,1,0,1,0,0,0,1], order="asc")
    assert galois.BCH(15, 5, primitive_poly=p).generator_poly == galois.Poly([1,0,1,0,0,1,1,0,1,1,1], order="asc")

    p = galois.Poly.Degrees([5, 4, 3, 2, 0])  # galois.primitive_poly(2, 5, method="max")
    assert galois.BCH(31, 26, primitive_poly=p).generator_poly == galois.Poly([1,0,1,1,1,1], order="asc")
    assert galois.BCH(31, 21, primitive_poly=p).generator_poly == galois.Poly([1,1,0,0,0,0,1,1,0,0,1], order="asc")
    assert galois.BCH(31, 16, primitive_poly=p).generator_poly == galois.Poly([1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1], order="asc")
    assert galois.BCH(31, 11, primitive_poly=p).generator_poly == galois.Poly([1,0,1,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1], order="asc")
    assert galois.BCH(31, 6, primitive_poly=p).generator_poly == galois.Poly([1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,1,1,1,1,0, 0,1,1,0,1,1], order="asc")

    p = galois.Poly.Degrees([6, 5, 4, 1, 0])  # galois.primitive_poly(2, 6, method="max")
    assert galois.BCH(63, 57, primitive_poly=p).generator_poly == galois.Poly([1,1,0,0,1,1,1], order="asc")


def test_bch_parity_check_matrix():
    # S. Lin and D. Costello. Error Control Coding. Example 6.2, p. 202.
    p = galois.Poly.Degrees([4,1,0])
    GF = galois.GF(2**4, irreducible_poly=p)
    alpha = GF.primitive_element
    bch = galois.BCH(15, 7)
    H_truth = alpha**np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
    ])
    assert np.array_equal(bch.H, np.fliplr(H_truth))  # NOTE: We use the convention of polynomial highest degree first, not last
