"""
A pytest module to test vector space operations over Galois fields.
"""

import numpy as np

import galois


def test_poly_evaluate_companion_matrix_is_zero():
    p, m = 3, 4
    poly = galois.irreducible_poly(p, m, method="random")
    C = galois.companion_matrix(poly)
    assert np.all(poly(C, elementwise=False) == 0)


def test_characteristic_and_minimal_poly():
    p, m = 3, 4
    poly = galois.irreducible_poly(p, m, method="random")
    C = galois.companion_matrix(poly)
    assert C.characteristic_poly() == poly
    assert C.minimal_poly() == poly


def test_multiplication_by_alpha():
    p, m = 3, 4
    poly = galois.irreducible_poly(p, m, method="random")
    GF = galois.GF(p**m, irreducible_poly=poly)
    x = GF.Random(10)
    y = x * GF("a")  # Multiply by alpha (the root of the irreducible polynomial)

    C = galois.companion_matrix(poly)  # Multiply by alpha matrix
    x_vec = x.vector()
    y_vec = x_vec @ C

    assert np.array_equal(y.vector(), y_vec)
    assert np.array_equal(y, GF.Vector(y_vec))


def test_linear_recurrent_sequence():
    p, m = 2, 5

    characteristic_poly = galois.primitive_poly(p, m, method="random")  # Characteristic polynomial of the LRS
    feedback_poly = characteristic_poly.reverse()  # Feedback polynomial of the LRS
    C = galois.companion_matrix(characteristic_poly)

    lfsr = galois.FLFSR(feedback_poly)
    length = 10
    outputs = lfsr.step(length)

    for i in range(0, length - m - 1):
        prev_m = outputs[i : i + m]
        next_m = outputs[i + 1 : i + 1 + m]
        # Order C to ascendning power basis for the LFSR
        assert np.array_equal(next_m, C[::-1, ::-1] @ prev_m)
        # Ordered next and prev arrays as descending power basis, so reverse the vectors
        assert np.array_equal(next_m[::-1], C @ prev_m[::-1])
