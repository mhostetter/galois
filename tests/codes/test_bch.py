"""
A pytest module to test BCH codes.

Test vectors generated from Octave with bchpoly().

References
----------
* https://octave.sourceforge.io/communications/function/bchpoly.html
"""
import random

import pytest
import numpy as np

import galois


def test_bch_valid_codes():
    """
    Generated in Octave with `bchpoly()`.
    """
    codes = np.array([
        [7, 4, 1],
    ])
    assert np.array_equal(galois.bch_valid_codes(7), codes)

    codes = np.array([
        [15,11, 1],
        [15, 7, 2],
        [15, 5, 3],
    ])
    assert np.array_equal(galois.bch_valid_codes(15), codes)

    codes = np.array([
        [31,26, 1],
        [31,21, 2],
        [31,16, 3],
        [31,11, 5],
        [31, 6, 7],
    ])
    assert np.array_equal(galois.bch_valid_codes(31), codes)

    codes = np.array([
        [63,57, 1],
        [63,51, 2],
        [63,45, 3],
        [63,39, 4],
        [63,36, 5],
        [63,30, 6],
        [63,24, 7],
        [63,18,10],
        [63,16,11],
        [63,10,13],
        [63, 7,15],
    ])
    assert np.array_equal(galois.bch_valid_codes(63), codes)

    codes = np.array([
        [127,120,  1],
        [127,113,  2],
        [127,106,  3],
        [127, 99,  4],
        [127, 92,  5],
        [127, 85,  6],
        [127, 78,  7],
        [127, 71,  9],
        [127, 64, 10],
        [127, 57, 11],
        [127, 50, 13],
        [127, 43, 14],
        [127, 36, 15],
        [127, 29, 21],
        [127, 22, 23],
        [127, 15, 27],
        [127,  8, 31],
    ])
    assert np.array_equal(galois.bch_valid_codes(127), codes)

    codes = np.array([
        [255,247,  1],
        [255,239,  2],
        [255,231,  3],
        [255,223,  4],
        [255,215,  5],
        [255,207,  6],
        [255,199,  7],
        [255,191,  8],
        [255,187,  9],
        [255,179, 10],
        [255,171, 11],
        [255,163, 12],
        [255,155, 13],
        [255,147, 14],
        [255,139, 15],
        [255,131, 18],
        [255,123, 19],
        [255,115, 21],
        [255,107, 22],
        [255, 99, 23],
        [255, 91, 25],
        [255, 87, 26],
        [255, 79, 27],
        [255, 71, 29],
        [255, 63, 30],
        [255, 55, 31],
        [255, 47, 42],
        [255, 45, 43],
        [255, 37, 45],
        [255, 29, 47],
        [255, 21, 55],
        [255, 13, 59],
        [255,  9, 63],
    ])
    assert np.array_equal(galois.bch_valid_codes(255), codes)

    codes = np.array([
        [511,502,  1],
        [511,493,  2],
        [511,484,  3],
        [511,475,  4],
        [511,466,  5],
        [511,457,  6],
        [511,448,  7],
        [511,439,  8],
        [511,430,  9],
        [511,421, 10],
        [511,412, 11],
        [511,403, 12],
        [511,394, 13],
        [511,385, 14],
        [511,376, 15],
        [511,367, 17],
        [511,358, 18],
        [511,349, 19],
        [511,340, 20],
        [511,331, 21],
        [511,322, 22],
        [511,313, 23],
        [511,304, 25],
        [511,295, 26],
        [511,286, 27],
        [511,277, 28],
        [511,268, 29],
        [511,259, 30],
        [511,250, 31],
        [511,241, 36],
        [511,238, 37],
        [511,229, 38],
        [511,220, 39],
        [511,211, 41],
        [511,202, 42],
        [511,193, 43],
        [511,184, 45],
        [511,175, 46],
        [511,166, 47],
        [511,157, 51],
        [511,148, 53],
        [511,139, 54],
        [511,130, 55],
        [511,121, 58],
        [511,112, 59],
        [511,103, 61],
        [511, 94, 62],
        [511, 85, 63],
        [511, 76, 85],
        [511, 67, 87],
        [511, 58, 91],
        [511, 49, 93],
        [511, 40, 95],
        [511, 31,109],
        [511, 28,111],
        [511, 19,119],
        [511, 10,127],
    ])
    assert np.array_equal(galois.bch_valid_codes(511), codes)


def test_bch_generator_poly():
    """
    S. Lin and D. Costello. Error Control Coding. Appendix C, pp. 1231.
    """
    assert galois.bch_generator_poly(7, 4).integer == 0o13

    assert galois.bch_generator_poly(15, 11).integer == 0o23
    assert galois.bch_generator_poly(15, 7).integer == 0o721
    assert galois.bch_generator_poly(15, 5).integer == 0o2467

    assert galois.bch_generator_poly(31, 26).integer == 0o45
    assert galois.bch_generator_poly(31, 21).integer == 0o3551
    assert galois.bch_generator_poly(31, 16).integer == 0o107657
    assert galois.bch_generator_poly(31, 11).integer == 0o5423325
    assert galois.bch_generator_poly(31, 6).integer == 0o313365047

    assert galois.bch_generator_poly(63, 57).integer == 0o103
    assert galois.bch_generator_poly(63, 51).integer == 0o12471
    assert galois.bch_generator_poly(63, 45).integer == 0o1701317
    assert galois.bch_generator_poly(63, 39).integer == 0o166623567
    assert galois.bch_generator_poly(63, 36).integer == 0o1033500423
    assert galois.bch_generator_poly(63, 30).integer == 0o157464165547
    assert galois.bch_generator_poly(63, 24).integer == 0o17323260404441
    assert galois.bch_generator_poly(63, 18).integer == 0o1363026512351725
    assert galois.bch_generator_poly(63, 16).integer == 0o6331141367235453
    assert galois.bch_generator_poly(63, 10).integer == 0o472622305527250155
    assert galois.bch_generator_poly(63, 7).integer == 0o5231045543503271737

    assert galois.bch_generator_poly(1023, 1013).integer == 0o2011
    assert galois.bch_generator_poly(1023, 1003).integer == 0o4014167
    # ...
    # assert galois.bch_generator_poly(1023, 11).integer == 0o3435423242053413257500125205705563224605


def test_bch_properties():
    bch = galois.BCH(7, 4)
    assert (bch.n, bch.k, bch.t) == (7, 4, 1)

    bch = galois.BCH(15, 11)
    assert (bch.n, bch.k, bch.t) == (15, 11, 1)

    bch = galois.BCH(15, 7)
    assert (bch.n, bch.k, bch.t) == (15, 7, 2)

    bch = galois.BCH(15, 5)
    assert (bch.n, bch.k, bch.t) == (15, 5, 3)


def test_bch_generator_poly_diff_primitive_poly():
    """
    Test with primitive polynomial others than the default. Generated in Octave with `bchpoly()`.
    """
    p = galois.Poly.Degrees([3, 2, 0])  # galois.primitive_poly(2, 3, method="largest")
    assert galois.bch_generator_poly(7, 4, primitive_poly=p) == galois.Poly([1,0,1,1], order="asc")

    p = galois.Poly.Degrees([4, 3, 0])  # galois.primitive_poly(2, 4, method="largest")
    assert galois.bch_generator_poly(15, 11, primitive_poly=p) == galois.Poly([1,0,0,1,1], order="asc")
    assert galois.bch_generator_poly(15, 7, primitive_poly=p) == galois.Poly([1,1,1,0,1,0,0,0,1], order="asc")
    assert galois.bch_generator_poly(15, 5, primitive_poly=p) == galois.Poly([1,0,1,0,0,1,1,0,1,1,1], order="asc")

    p = galois.Poly.Degrees([5, 4, 3, 2, 0])  # galois.primitive_poly(2, 5, method="largest")
    assert galois.bch_generator_poly(31, 26, primitive_poly=p) == galois.Poly([1,0,1,1,1,1], order="asc")
    assert galois.bch_generator_poly(31, 21, primitive_poly=p) == galois.Poly([1,1,0,0,0,0,1,1,0,0,1], order="asc")
    assert galois.bch_generator_poly(31, 16, primitive_poly=p) == galois.Poly([1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1], order="asc")
    assert galois.bch_generator_poly(31, 11, primitive_poly=p) == galois.Poly([1,0,1,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1], order="asc")
    assert galois.bch_generator_poly(31, 6, primitive_poly=p) == galois.Poly([1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,1,1,1,1,0, 0,1,1,0,1,1], order="asc")

    p = galois.Poly.Degrees([6, 5, 4, 1, 0])  # galois.primitive_poly(2, 6, method="largest")
    assert galois.bch_generator_poly(63, 57, primitive_poly=p) == galois.Poly([1,1,0,0,1,1,1], order="asc")
