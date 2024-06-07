"""
A pytest module to test general BCH codes.
"""

import random

import numpy as np
import pytest

import galois

from .conftest import (
    verify_decode,
    verify_decode_shortened,
    verify_encode,
    verify_encode_shortened,
)


def test_exceptions():
    with pytest.raises(TypeError):
        galois.BCH(15.0, 7)
    with pytest.raises(TypeError):
        galois.BCH(15, 7.0)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, c=1.0)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, field=2)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, extension_field=2**4)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, alpha=2.0)
    with pytest.raises(TypeError):
        galois.BCH(15, 7, systematic=1)

    with pytest.raises(ValueError):
        galois.BCH(15, 12)
    with pytest.raises(ValueError):
        galois.BCH(14, 7)
    with pytest.raises(ValueError):
        galois.BCH(15, 12, 3)
    with pytest.raises(ValueError):
        galois.BCH(15, 11, 4)
    with pytest.raises(ValueError):
        galois.BCH(15, 7, field=galois.GF(2**2))


def test_repr():
    bch = galois.BCH(15, 7)
    assert repr(bch) == "<BCH Code: [15, 7, 5] over GF(2)>"


def test_str():
    bch = galois.BCH(15, 7)
    assert (
        str(bch)
        == "BCH Code:\n  [n, k, d]: [15, 7, 5]\n  field: GF(2)\n  extension_field: GF(2^4)\n  generator_poly: x^8 + x^7 + x^6 + x^4 + 1\n  is_primitive: True\n  is_narrow_sense: True\n  is_systematic: True"
    )


def test_properties(bch_codes):
    bch = bch_codes["code"]

    assert bch.n == bch_codes["n"]
    assert bch.k == bch_codes["k"]
    assert bch.d == bch_codes["d"]
    # assert bch.d_min == bch_codes["d_min"]

    assert bch.is_systematic == bch_codes["is_systematic"]
    assert bch.is_primitive == bch_codes["is_primitive"]
    assert bch.is_narrow_sense == bch_codes["is_narrow_sense"]

    assert isinstance(bch.alpha, bch.extension_field)
    assert bch.alpha == bch_codes["alpha"]

    assert bch.c == bch_codes["c"]

    assert isinstance(bch.generator_poly, galois.Poly)
    assert bch.generator_poly == bch_codes["generator_poly"]
    assert bch.generator_poly.field is bch.field

    # Test roots are zeros of the generator polynomial in the extension field
    assert np.all(galois.Poly(bch.generator_poly.coeffs, field=bch.extension_field)(bch.roots) == 0)

    assert isinstance(bch.G, bch.field)
    assert np.array_equal(bch.G, bch_codes["G"])

    assert isinstance(bch.parity_check_poly, galois.Poly)
    assert bch.parity_check_poly == bch_codes["parity_check_poly"]
    assert bch.parity_check_poly.field is bch.field

    assert isinstance(bch.H, bch.field)
    assert np.array_equal(bch.H, bch_codes["H"])


def test_encode_exceptions():
    # Systematic
    n, k = 15, 7
    bch = galois.BCH(n, k)
    GF = bch.field
    with pytest.raises(ValueError):
        bch.encode(GF.Random(k + 1))

    # Non-systematic
    n, k = 15, 7
    bch = galois.BCH(n, k, systematic=False)
    GF = bch.field
    with pytest.raises(ValueError):
        bch.encode(GF.Random(k - 1))


def test_encode_vector(bch_codes):
    if bch_codes["d"] == 1:
        return
    bch = bch_codes["code"]
    MESSAGES = bch_codes["encode"]["messages"]
    CODEWORDS = bch_codes["encode"]["codewords"]
    is_systematic = bch_codes["is_systematic"]

    verify_encode(bch, MESSAGES, CODEWORDS, is_systematic, True)


def test_encode_matrix(bch_codes):
    if bch_codes["d"] == 1:
        return
    bch = bch_codes["code"]
    MESSAGES = bch_codes["encode"]["messages"]
    CODEWORDS = bch_codes["encode"]["codewords"]
    is_systematic = bch_codes["is_systematic"]

    verify_encode(bch, MESSAGES, CODEWORDS, is_systematic, False)


def test_encode_shortened_vector(bch_codes):
    if bch_codes["d"] == 1:
        return
    bch = bch_codes["code"]
    MESSAGES = bch_codes["encode"]["messages"]
    CODEWORDS = bch_codes["encode"]["codewords"]
    is_systematic = bch_codes["is_systematic"]

    verify_encode_shortened(bch, MESSAGES, CODEWORDS, is_systematic, True)


def test_encode_shortened_matrix(bch_codes):
    bch = bch_codes["code"]
    MESSAGES = bch_codes["encode"]["messages"]
    CODEWORDS = bch_codes["encode"]["codewords"]
    is_systematic = bch_codes["is_systematic"]

    verify_encode_shortened(bch, MESSAGES, CODEWORDS, is_systematic, False)


def test_decode_exceptions():
    # Systematic
    n, k = 15, 7
    bch = galois.BCH(n, k)
    GF = galois.GF2
    with pytest.raises(ValueError):
        bch.decode(GF.Random(n + 1))

    # Non-systematic
    n, k = 15, 7
    bch = galois.BCH(n, k, systematic=False)
    GF = galois.GF2
    with pytest.raises(ValueError):
        bch.decode(GF.Random(n - 1))


def test_decode_vector(bch_codes):
    bch = bch_codes["code"]
    verify_decode(bch, 1)


def test_decode_matrix(bch_codes):
    bch = bch_codes["code"]
    verify_decode(bch, 5)


def test_decode_shortened_vector(bch_codes):
    bch = bch_codes["code"]
    is_systematic = bch_codes["is_systematic"]
    verify_decode_shortened(bch, 1, is_systematic)


def test_decode_shortened_matrix(bch_codes):
    bch = bch_codes["code"]
    is_systematic = bch_codes["is_systematic"]
    verify_decode_shortened(bch, 5, is_systematic)


def test_bch_valid_codes_7():
    codes = [
        (7, 4, 1),
        (7, 1, 3),
    ]
    code = random.choice(codes)
    bch = galois.BCH(code[0], code[1])
    assert (bch.n, bch.k, bch.t) == code


def test_bch_valid_codes_15():
    codes = [
        (15, 11, 1),
        (15, 7, 2),
        (15, 5, 3),
        (15, 1, 7),
    ]
    code = random.choice(codes)
    bch = galois.BCH(code[0], code[1])
    assert (bch.n, bch.k, bch.t) == code


def test_bch_valid_codes_31():
    codes = [
        (31, 26, 1),
        (31, 21, 2),
        (31, 16, 3),
        (31, 11, 5),
        (31, 6, 7),
        (31, 1, 15),
    ]
    code = random.choice(codes)
    bch = galois.BCH(code[0], code[1])
    assert (bch.n, bch.k, bch.t) == code


def test_bch_valid_codes_63():
    codes = [
        (63, 57, 1),
        (63, 51, 2),
        (63, 45, 3),
        (63, 39, 4),
        (63, 36, 5),
        (63, 30, 6),
        (63, 24, 7),
        (63, 18, 10),
        (63, 16, 11),
        (63, 10, 13),
        (63, 7, 15),
        (63, 1, 31),
    ]
    code = random.choice(codes)
    bch = galois.BCH(code[0], code[1])
    assert (bch.n, bch.k, bch.t) == code


def test_bch_valid_codes_127():
    codes = [
        (127, 120, 1),
        (127, 113, 2),
        (127, 106, 3),
        (127, 99, 4),
        (127, 92, 5),
        (127, 85, 6),
        (127, 78, 7),
        (127, 71, 9),
        (127, 64, 10),
        (127, 57, 11),
        (127, 50, 13),
        (127, 43, 14),
        (127, 36, 15),
        (127, 29, 21),
        (127, 22, 23),
        (127, 15, 27),
        (127, 8, 31),
        (127, 1, 63),
    ]
    code = random.choice(codes)
    bch = galois.BCH(code[0], code[1])
    assert (bch.n, bch.k, bch.t) == code


def test_bch_valid_codes_255():
    codes = [
        (255, 247, 1),
        (255, 239, 2),
        (255, 231, 3),
        (255, 223, 4),
        (255, 215, 5),
        (255, 207, 6),
        (255, 199, 7),
        (255, 191, 8),
        (255, 187, 9),
        (255, 179, 10),
        (255, 171, 11),
        (255, 163, 12),
        (255, 155, 13),
        (255, 147, 14),
        (255, 139, 15),
        (255, 131, 18),
        (255, 123, 19),
        (255, 115, 21),
        (255, 107, 22),
        (255, 99, 23),
        (255, 91, 25),
        (255, 87, 26),
        (255, 79, 27),
        (255, 71, 29),
        (255, 63, 30),
        (255, 55, 31),
        (255, 47, 42),
        (255, 45, 43),
        (255, 37, 45),
        (255, 29, 47),
        (255, 21, 55),
        (255, 13, 59),
        (255, 9, 63),
        (255, 1, 127),
    ]
    code = random.choice(codes)
    bch = galois.BCH(code[0], code[1])
    assert (bch.n, bch.k, bch.t) == code


def test_bch_valid_codes_511():
    codes = [
        (511, 502, 1),
        (511, 493, 2),
        (511, 484, 3),
        (511, 475, 4),
        (511, 466, 5),
        (511, 457, 6),
        (511, 448, 7),
        (511, 439, 8),
        (511, 430, 9),
        (511, 421, 10),
        (511, 412, 11),
        (511, 403, 12),
        (511, 394, 13),
        (511, 385, 14),
        (511, 376, 15),
        (511, 367, 17),
        (511, 358, 18),
        (511, 349, 19),
        (511, 340, 20),
        (511, 331, 21),
        (511, 322, 22),
        (511, 313, 23),
        (511, 304, 25),
        (511, 295, 26),
        (511, 286, 27),
        (511, 277, 28),
        (511, 268, 29),
        (511, 259, 30),
        (511, 250, 31),
        (511, 241, 36),
        (511, 238, 37),
        (511, 229, 38),
        (511, 220, 39),
        (511, 211, 41),
        (511, 202, 42),
        (511, 193, 43),
        (511, 184, 45),
        (511, 175, 46),
        (511, 166, 47),
        (511, 157, 51),
        (511, 148, 53),
        (511, 139, 54),
        (511, 130, 55),
        (511, 121, 58),
        (511, 112, 59),
        (511, 103, 61),
        (511, 94, 62),
        (511, 85, 63),
        (511, 76, 85),
        (511, 67, 87),
        (511, 58, 91),
        (511, 49, 93),
        (511, 40, 95),
        (511, 31, 109),
        (511, 28, 111),
        (511, 19, 119),
        (511, 10, 127),
        (511, 1, 255),
    ]
    code = random.choice(codes)
    bch = galois.BCH(code[0], code[1])
    assert (bch.n, bch.k, bch.t) == code


def test_bug_483():
    """
    See https://github.com/mhostetter/galois/issues/483.
    """
    bch_1 = galois.BCH(15, 11)
    verify_decode(bch_1, 1)

    bch_2 = galois.BCH(7, 4)
    verify_decode(bch_2, 1)

    bch_3 = galois.BCH(31, 26)
    verify_decode(bch_3, 1)
