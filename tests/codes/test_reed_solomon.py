"""
A pytest module to test general Reed-Solomon codes.
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
        galois.ReedSolomon(15.0, 7)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 7.0)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 7, d=9.0)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 7, field=2**4)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 7, alpha=2.0)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 7, c=1.0)
    with pytest.raises(TypeError):
        galois.ReedSolomon(15, 7, systematic=1)

    with pytest.raises(ValueError):
        galois.ReedSolomon(14, 7)
    with pytest.raises(ValueError):
        galois.ReedSolomon(15, 12, 3)
    with pytest.raises(ValueError):
        galois.ReedSolomon(15, 11, 4)
    with pytest.raises(ValueError):
        galois.ReedSolomon(15, 7, field=galois.GF(2**2))


def test_repr():
    rs = galois.ReedSolomon(15, 11)
    assert repr(rs) == "<Reed-Solomon Code: [15, 11, 5] over GF(2^4)>"


def test_str():
    rs = galois.ReedSolomon(15, 11)
    assert (
        str(rs)
        == "Reed-Solomon Code:\n  [n, k, d]: [15, 11, 5]\n  field: GF(2^4)\n  generator_poly: x^4 + 13x^3 + 12x^2 + 8x + 7\n  is_primitive: True\n  is_narrow_sense: True\n  is_systematic: True"
    )


def test_properties(reed_solomon_codes):
    rs = reed_solomon_codes["code"]

    assert rs.n == reed_solomon_codes["n"]
    assert rs.k == reed_solomon_codes["k"]
    assert rs.d == reed_solomon_codes["d"]

    assert rs.is_systematic == reed_solomon_codes["is_systematic"]
    assert rs.is_primitive == reed_solomon_codes["is_primitive"]
    assert rs.is_narrow_sense == reed_solomon_codes["is_narrow_sense"]

    assert isinstance(rs.alpha, rs.field)
    assert rs.alpha == reed_solomon_codes["alpha"]

    assert rs.c == reed_solomon_codes["c"]

    assert isinstance(rs.generator_poly, galois.Poly)
    assert rs.generator_poly == reed_solomon_codes["generator_poly"]
    assert rs.generator_poly.field is rs.field

    # Test roots are zeros of the generator polynomial in the extension field
    assert np.all(rs.generator_poly(rs.roots) == 0)

    assert isinstance(rs.G, rs.field)
    assert np.array_equal(rs.G, reed_solomon_codes["G"])

    assert isinstance(rs.H, rs.field)
    assert np.array_equal(rs.H, reed_solomon_codes["H"])


@pytest.mark.parametrize("is_systematic", (True, False))
def test_encode_exceptions(is_systematic):
    n, k = 15, 7
    rs = galois.ReedSolomon(n, k, systematic=is_systematic)
    GF = rs.field
    with pytest.raises(ValueError):
        rs.encode(GF.Random(k + 1))


def test_encode_vector(reed_solomon_codes):
    if reed_solomon_codes["d"] == 1:
        return
    rs = reed_solomon_codes["code"]
    MESSAGES = reed_solomon_codes["encode"]["messages"]
    CODEWORDS = reed_solomon_codes["encode"]["codewords"]

    verify_encode(rs, MESSAGES, CODEWORDS, True)


def test_encode_matrix(reed_solomon_codes):
    if reed_solomon_codes["d"] == 1:
        return
    rs = reed_solomon_codes["code"]
    MESSAGES = reed_solomon_codes["encode"]["messages"]
    CODEWORDS = reed_solomon_codes["encode"]["codewords"]

    verify_encode(rs, MESSAGES, CODEWORDS, False)


def test_encode_shortened_vector(reed_solomon_codes):
    if reed_solomon_codes["d"] == 1:
        return
    rs = reed_solomon_codes["code"]
    MESSAGES = reed_solomon_codes["encode"]["messages"]
    CODEWORDS = reed_solomon_codes["encode"]["codewords"]

    verify_encode_shortened(rs, MESSAGES, CODEWORDS, True)


def test_encode_shortened_matrix(reed_solomon_codes):
    rs = reed_solomon_codes["code"]
    MESSAGES = reed_solomon_codes["encode"]["messages"]
    CODEWORDS = reed_solomon_codes["encode"]["codewords"]

    verify_encode_shortened(rs, MESSAGES, CODEWORDS, False)


@pytest.mark.parametrize("is_systematic", (True, False))
def test_decode_exceptions(is_systematic):
    n, k = 15, 7
    rs = galois.ReedSolomon(n, k, systematic=is_systematic)
    GF = rs.field
    with pytest.raises(ValueError):
        rs.decode(GF.Random(n + 1))


def test_decode_vector(reed_solomon_codes):
    rs = reed_solomon_codes["code"]
    verify_decode(rs, 1)


def test_decode_matrix(reed_solomon_codes):
    rs = reed_solomon_codes["code"]
    verify_decode(rs, 5)


def test_decode_shortened_vector(reed_solomon_codes):
    rs = reed_solomon_codes["code"]
    verify_decode_shortened(rs, 1)


def test_decode_shortened_matrix(reed_solomon_codes):
    rs = reed_solomon_codes["code"]
    verify_decode_shortened(rs, 5)


def test_odd_characteristic():
    rs = galois.ReedSolomon(3**2 - 1, d=3, field=galois.GF(3**2))
    message = rs.field.Range(0, rs.k)
    codeword = rs.encode(message)
    err_codeword = codeword.copy()
    err_codeword[0] += rs.field(1)
    decoded_message, num_errors = rs.decode(err_codeword, errors=True)
    assert num_errors == 1
    assert np.array_equal(decoded_message, message)


@pytest.mark.parametrize("q", [2**4, 3**3])
def test_errors_and_erasures(q):
    rs = galois.ReedSolomon(q - 1, d=7, field=galois.GF(q))
    message = rs.field.Random(rs.k)
    codeword = rs.encode(message)

    for n_erasures in range(1, rs.d):
        c = codeword.copy()

        # Add erasures
        erasure_idxs = np.arange(n_erasures)
        erasure_idxs = random.sample(erasure_idxs.tolist(), k=n_erasures)
        erasures = np.zeros(codeword.shape, dtype=bool)  # Erasure mask
        erasures[erasure_idxs] = True
        c[erasures] = 0  # Erasures are represented by zeros

        # Add a correctable number of errors
        n_errors = (rs.d - 1 - n_erasures) // 2
        error_idxs = np.where(~erasures)[0]  # Possible error indices
        error_idxs = random.sample(error_idxs.tolist(), k=n_errors)
        errors = np.zeros(codeword.shape, dtype=bool)  # Error mask
        errors[error_idxs] = True
        c[errors] += rs.field.Random(1, low=1)  # Introduce errors

        decoded_message, n_corrected = rs.decode(c, erasures=erasures, errors=True)
        assert np.array_equal(decoded_message, message)
        assert n_corrected == n_errors


@pytest.mark.parametrize("q", [2**4, 3**3])
def test_errors_and_erasures_shortened(q):
    rs = galois.ReedSolomon(q - 1, d=7, field=galois.GF(q))
    s = 3  # Shortening length
    message = rs.field.Random(rs.k - s)
    codeword = rs.encode(message)

    for n_erasures in range(1, rs.d):
        c = codeword.copy()

        # Add erasures
        erasure_idxs = np.arange(n_erasures)
        erasure_idxs = random.sample(erasure_idxs.tolist(), k=n_erasures)
        erasures = np.zeros(codeword.shape, dtype=bool)  # Erasure mask
        erasures[erasure_idxs] = True
        c[erasures] = 0  # Erasures are represented by zeros

        # Add a correctable number of errors
        n_errors = (rs.d - 1 - n_erasures) // 2
        error_idxs = np.where(~erasures)[0]  # Possible error indices
        error_idxs = random.sample(error_idxs.tolist(), k=n_errors)
        errors = np.zeros(codeword.shape, dtype=bool)  # Error mask
        errors[error_idxs] = True
        c[errors] += rs.field.Random(1, low=1)  # Introduce errors

        decoded_message, n_corrected = rs.decode(c, erasures=erasures, errors=True)
        assert np.array_equal(decoded_message, message)
        assert n_corrected == n_errors
