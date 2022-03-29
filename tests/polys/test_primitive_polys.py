"""
A pytest module to test generating primitive polynomials and testing primitivity.

Sage:
    def integer(coeffs, order):
        i = 0
        for d, c in enumerate(coeffs[::-1]):
            i += (c.integer_representation() * order**d)
        return i

    PARAMS = [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4), (2**2,1), (2**2,2), (2**2,3), (3**2,1), (3**2,2), (3**2,3), (5**2,1), (5**2,2)]
    for order, degree in PARAMS:
        list_ = []
        R = GF(order, repr="int")["x"]
        for f in R.polynomials(degree):
            # For some reason `is_primitive()` crashes on f(x) = x
            if f.coefficients(sparse=False) == [0, 1]:
                continue
            if f.is_monic() and f.is_primitive():
                list_.append(f.coefficients(sparse=False)[::-1])

        # Sort in lexicographically-increasing order
        if not is_prime(order):
            list_ = sorted(list_, key=lambda item: integer(item, order))

        print(f"PRIMITIVE_POLYS_{order}_{degree} = {list_}")

References
----------
* https://baylor-ir.tdl.org/bitstream/handle/2104/8793/GF3%20Polynomials.pdf?sequence=1&isAllowed=y
"""
import pytest

import galois

PARAMS = [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4), (2**2,1), (2**2,2), (2**2,3), (3**2,1), (3**2,2), (3**2,3), (5**2,1), (5**2,2)]

# LUT items are poly coefficients in degree-descending order

PRIMITIVE_POLYS_2_1 = [[1, 1]]
PRIMITIVE_POLYS_2_2 = [[1, 1, 1]]
PRIMITIVE_POLYS_2_3 = [[1, 0, 1, 1], [1, 1, 0, 1]]
PRIMITIVE_POLYS_2_4 = [[1, 0, 0, 1, 1], [1, 1, 0, 0, 1]]
PRIMITIVE_POLYS_2_5 = [[1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1]]
PRIMITIVE_POLYS_2_6 = [[1, 0, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 0, 1, 1]]
PRIMITIVE_POLYS_2_7 = [[1, 0, 0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 1, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 0, 0, 1], [1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1], [1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 1]]
PRIMITIVE_POLYS_2_8 = [[1, 0, 0, 0, 1, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 0, 1]]

PRIMITIVE_POLYS_3_1 = [[1, 1]]
PRIMITIVE_POLYS_3_2 = [[1, 1, 2], [1, 2, 2]]
PRIMITIVE_POLYS_3_3 = [[1, 0, 2, 1], [1, 1, 2, 1], [1, 2, 0, 1], [1, 2, 1, 1]]
PRIMITIVE_POLYS_3_4 = [[1, 0, 0, 1, 2], [1, 0, 0, 2, 2], [1, 1, 0, 0, 2], [1, 1, 1, 2, 2], [1, 1, 2, 2, 2], [1, 2, 0, 0, 2], [1, 2, 1, 1, 2], [1, 2, 2, 1, 2]]
PRIMITIVE_POLYS_3_5 = [[1, 0, 0, 0, 2, 1], [1, 0, 0, 2, 1, 1], [1, 0, 1, 0, 1, 1], [1, 0, 1, 2, 0, 1], [1, 0, 1, 2, 2, 1], [1, 0, 2, 1, 0, 1], [1, 0, 2, 2, 1, 1], [1, 1, 0, 0, 2, 1], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 2, 1, 1], [1, 1, 2, 0, 0, 1], [1, 1, 2, 1, 1, 1], [1, 1, 2, 2, 0, 1], [1, 2, 0, 0, 0, 1], [1, 2, 0, 0, 1, 1], [1, 2, 0, 2, 2, 1], [1, 2, 1, 1, 1, 1], [1, 2, 2, 0, 2, 1], [1, 2, 2, 1, 0, 1]]
PRIMITIVE_POLYS_3_6 = [[1, 0, 0, 0, 0, 1, 2], [1, 0, 0, 0, 0, 2, 2], [1, 0, 0, 1, 0, 1, 2], [1, 0, 0, 2, 0, 2, 2], [1, 0, 1, 0, 2, 1, 2], [1, 0, 1, 0, 2, 2, 2], [1, 0, 1, 1, 0, 2, 2], [1, 0, 1, 1, 1, 2, 2], [1, 0, 1, 2, 0, 1, 2], [1, 0, 1, 2, 1, 1, 2], [1, 0, 2, 0, 1, 1, 2], [1, 0, 2, 0, 1, 2, 2], [1, 0, 2, 1, 1, 1, 2], [1, 0, 2, 2, 1, 2, 2], [1, 1, 0, 0, 0, 0, 2], [1, 1, 0, 1, 0, 0, 2], [1, 1, 0, 1, 1, 1, 2], [1, 1, 0, 1, 2, 1, 2], [1, 1, 0, 2, 2, 0, 2], [1, 1, 1, 0, 1, 2, 2], [1, 1, 1, 0, 2, 0, 2], [1, 1, 1, 1, 0, 1, 2], [1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2, 2], [1, 1, 2, 0, 1, 0, 2], [1, 1, 2, 0, 2, 2, 2], [1, 1, 2, 1, 0, 1, 2], [1, 1, 2, 1, 1, 0, 2], [1, 1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 2, 0, 2], [1, 2, 0, 0, 0, 0, 2], [1, 2, 0, 1, 2, 0, 2], [1, 2, 0, 2, 0, 0, 2], [1, 2, 0, 2, 1, 2, 2], [1, 2, 0, 2, 2, 2, 2], [1, 2, 1, 0, 1, 1, 2], [1, 2, 1, 0, 2, 0, 2], [1, 2, 1, 1, 2, 1, 2], [1, 2, 1, 2, 0, 2, 2], [1, 2, 1, 2, 1, 2, 2], [1, 2, 1, 2, 2, 1, 2], [1, 2, 2, 0, 1, 0, 2], [1, 2, 2, 0, 2, 1, 2], [1, 2, 2, 1, 2, 0, 2], [1, 2, 2, 2, 0, 2, 2], [1, 2, 2, 2, 1, 0, 2], [1, 2, 2, 2, 2, 2, 2]]

PRIMITIVE_POLYS_5_1 = [[1, 2], [1, 3]]
PRIMITIVE_POLYS_5_2 = [[1, 1, 2], [1, 2, 3], [1, 3, 3], [1, 4, 2]]
PRIMITIVE_POLYS_5_3 = [[1, 0, 3, 2], [1, 0, 3, 3], [1, 0, 4, 2], [1, 0, 4, 3], [1, 1, 0, 2], [1, 1, 1, 3], [1, 1, 4, 3], [1, 2, 0, 3], [1, 2, 1, 3], [1, 2, 2, 2], [1, 2, 2, 3], [1, 2, 4, 2], [1, 3, 0, 2], [1, 3, 1, 2], [1, 3, 2, 2], [1, 3, 2, 3], [1, 3, 4, 3], [1, 4, 0, 3], [1, 4, 1, 2], [1, 4, 4, 2]]
PRIMITIVE_POLYS_5_4 = [[1, 0, 1, 2, 2], [1, 0, 1, 2, 3], [1, 0, 1, 3, 2], [1, 0, 1, 3, 3], [1, 0, 4, 1, 2], [1, 0, 4, 1, 3], [1, 0, 4, 4, 2], [1, 0, 4, 4, 3], [1, 1, 0, 1, 3], [1, 1, 0, 2, 3], [1, 1, 0, 3, 2], [1, 1, 0, 4, 2], [1, 1, 1, 1, 3], [1, 1, 2, 0, 2], [1, 1, 2, 1, 2], [1, 1, 3, 0, 3], [1, 1, 3, 4, 2], [1, 1, 4, 4, 3], [1, 2, 0, 1, 3], [1, 2, 0, 2, 2], [1, 2, 0, 3, 3], [1, 2, 0, 4, 2], [1, 2, 1, 2, 3], [1, 2, 2, 0, 3], [1, 2, 2, 2, 2], [1, 2, 3, 0, 2], [1, 2, 3, 3, 2], [1, 2, 4, 3, 3], [1, 3, 0, 1, 2], [1, 3, 0, 2, 3], [1, 3, 0, 3, 2], [1, 3, 0, 4, 3], [1, 3, 1, 3, 3], [1, 3, 2, 0, 3], [1, 3, 2, 3, 2], [1, 3, 3, 0, 2], [1, 3, 3, 2, 2], [1, 3, 4, 2, 3], [1, 4, 0, 1, 2], [1, 4, 0, 2, 2], [1, 4, 0, 3, 3], [1, 4, 0, 4, 3], [1, 4, 1, 4, 3], [1, 4, 2, 0, 2], [1, 4, 2, 4, 2], [1, 4, 3, 0, 3], [1, 4, 3, 1, 2], [1, 4, 4, 1, 3]]

PRIMITIVE_POLYS_4_1 = [[1, 2], [1, 3]]
PRIMITIVE_POLYS_4_2 = [[1, 1, 2], [1, 1, 3], [1, 2, 2], [1, 3, 3]]
PRIMITIVE_POLYS_4_3 = [[1, 1, 1, 2], [1, 1, 1, 3], [1, 1, 2, 3], [1, 1, 3, 2], [1, 2, 1, 3], [1, 2, 2, 2], [1, 2, 3, 2], [1, 2, 3, 3], [1, 3, 1, 2], [1, 3, 2, 2], [1, 3, 2, 3], [1, 3, 3, 3]]

PRIMITIVE_POLYS_9_1 = [[1, 3], [1, 5], [1, 6], [1, 7]]
PRIMITIVE_POLYS_9_2 = [[1, 1, 3], [1, 1, 7], [1, 2, 3], [1, 2, 7], [1, 3, 6], [1, 3, 7], [1, 4, 5], [1, 4, 6], [1, 5, 3], [1, 5, 5], [1, 6, 6], [1, 6, 7], [1, 7, 3], [1, 7, 5], [1, 8, 5], [1, 8, 6]]
PRIMITIVE_POLYS_9_3 = [[1, 0, 1, 3], [1, 0, 1, 5], [1, 0, 1, 6], [1, 0, 1, 7], [1, 0, 2, 3], [1, 0, 2, 5], [1, 0, 2, 6], [1, 0, 2, 7], [1, 1, 2, 3], [1, 1, 2, 7], [1, 1, 3, 5], [1, 1, 4, 5], [1, 1, 4, 7], [1, 1, 5, 3], [1, 1, 6, 7], [1, 1, 7, 6], [1, 1, 8, 3], [1, 1, 8, 6], [1, 2, 2, 5], [1, 2, 2, 6], [1, 2, 3, 7], [1, 2, 4, 5], [1, 2, 4, 7], [1, 2, 5, 6], [1, 2, 6, 5], [1, 2, 7, 3], [1, 2, 8, 3], [1, 2, 8, 6], [1, 3, 0, 3], [1, 3, 0, 6], [1, 3, 1, 5], [1, 3, 2, 5], [1, 3, 3, 7], [1, 3, 4, 3], [1, 3, 4, 6], [1, 3, 5, 7], [1, 3, 6, 6], [1, 3, 6, 7], [1, 3, 7, 3], [1, 3, 7, 7], [1, 4, 1, 3], [1, 4, 1, 5], [1, 4, 3, 3], [1, 4, 4, 5], [1, 4, 4, 7], [1, 4, 5, 7], [1, 4, 6, 6], [1, 4, 7, 5], [1, 4, 8, 3], [1, 4, 8, 6], [1, 5, 0, 5], [1, 5, 0, 7], [1, 5, 1, 3], [1, 5, 2, 3], [1, 5, 3, 5], [1, 5, 3, 6], [1, 5, 5, 6], [1, 5, 5, 7], [1, 5, 6, 6], [1, 5, 7, 6], [1, 5, 8, 5], [1, 5, 8, 7], [1, 6, 0, 3], [1, 6, 0, 6], [1, 6, 1, 7], [1, 6, 2, 7], [1, 6, 3, 5], [1, 6, 4, 3], [1, 6, 4, 6], [1, 6, 5, 5], [1, 6, 6, 3], [1, 6, 6, 5], [1, 6, 7, 5], [1, 6, 7, 6], [1, 7, 0, 5], [1, 7, 0, 7], [1, 7, 1, 6], [1, 7, 2, 6], [1, 7, 3, 3], [1, 7, 3, 7], [1, 7, 5, 3], [1, 7, 5, 5], [1, 7, 6, 3], [1, 7, 7, 3], [1, 7, 8, 5], [1, 7, 8, 7], [1, 8, 1, 6], [1, 8, 1, 7], [1, 8, 3, 6], [1, 8, 4, 5], [1, 8, 4, 7], [1, 8, 5, 5], [1, 8, 6, 3], [1, 8, 7, 7], [1, 8, 8, 3], [1, 8, 8, 6]]

PRIMITIVE_POLYS_25_1 = [[1, 5], [1, 9], [1, 10], [1, 13], [1, 15], [1, 17], [1, 20], [1, 21]]
PRIMITIVE_POLYS_25_2 = [[1, 1, 9], [1, 1, 13], [1, 1, 15], [1, 1, 20], [1, 2, 5], [1, 2, 10], [1, 2, 17], [1, 2, 21], [1, 3, 5], [1, 3, 10], [1, 3, 17], [1, 3, 21], [1, 4, 9], [1, 4, 13], [1, 4, 15], [1, 4, 20], [1, 5, 5], [1, 5, 13], [1, 5, 15], [1, 5, 17], [1, 6, 9], [1, 6, 15], [1, 6, 20], [1, 6, 21], [1, 7, 5], [1, 7, 13], [1, 7, 15], [1, 7, 21], [1, 8, 5], [1, 8, 9], [1, 8, 13], [1, 8, 20], [1, 9, 10], [1, 9, 13], [1, 9, 15], [1, 9, 21], [1, 10, 10], [1, 10, 13], [1, 10, 17], [1, 10, 20], [1, 11, 5], [1, 11, 17], [1, 11, 20], [1, 11, 21], [1, 12, 5], [1, 12, 9], [1, 12, 10], [1, 12, 21], [1, 13, 9], [1, 13, 10], [1, 13, 15], [1, 13, 17], [1, 14, 9], [1, 14, 10], [1, 14, 17], [1, 14, 20], [1, 15, 10], [1, 15, 13], [1, 15, 17], [1, 15, 20], [1, 16, 9], [1, 16, 10], [1, 16, 17], [1, 16, 20], [1, 17, 9], [1, 17, 10], [1, 17, 15], [1, 17, 17], [1, 18, 5], [1, 18, 9], [1, 18, 10], [1, 18, 21], [1, 19, 5], [1, 19, 17], [1, 19, 20], [1, 19, 21], [1, 20, 5], [1, 20, 13], [1, 20, 15], [1, 20, 17], [1, 21, 10], [1, 21, 13], [1, 21, 15], [1, 21, 21], [1, 22, 5], [1, 22, 9], [1, 22, 13], [1, 22, 20], [1, 23, 5], [1, 23, 13], [1, 23, 15], [1, 23, 21], [1, 24, 9], [1, 24, 15], [1, 24, 20], [1, 24, 21]]


def test_primitive_poly_exceptions():
    with pytest.raises(TypeError):
        galois.primitive_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.primitive_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.primitive_poly(2**2 * 3**2, 3)
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 0)
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 3, method="invalid-argument")


@pytest.mark.parametrize("order,degree", PARAMS)
def test_primitive_poly_min(order, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{order}_{degree}")
    assert galois.primitive_poly(order, degree).coeffs.tolist() == LUT[0]


@pytest.mark.parametrize("order,degree", PARAMS)
def test_primitive_poly_max(order, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{order}_{degree}")
    assert galois.primitive_poly(order, degree, method="max").coeffs.tolist() == LUT[-1]


@pytest.mark.parametrize("order,degree", PARAMS)
def test_primitive_poly_random(order, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{order}_{degree}")
    assert galois.primitive_poly(order, degree, method="random").coeffs.tolist() in LUT


def test_primitive_polys_exceptions():
    with pytest.raises(TypeError):
        next(galois.primitive_polys(2.0, 3))
    with pytest.raises(TypeError):
        next(galois.primitive_polys(2, 3.0))
    with pytest.raises(TypeError):
        next(galois.primitive_polys(2, 3, reverse=1))
    with pytest.raises(ValueError):
        next(galois.primitive_polys(2**2 * 3**2, 3))
    with pytest.raises(ValueError):
        next(galois.primitive_polys(2, -1))


@pytest.mark.parametrize("order,degree", PARAMS)
def test_primitive_polys(order, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{order}_{degree}")
    assert [f.coeffs.tolist() for f in galois.primitive_polys(order, degree)] == LUT


def test_conway_poly_exceptions():
    with pytest.raises(TypeError):
        galois.conway_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.conway_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.conway_poly(4, 3)
    with pytest.raises(ValueError):
        galois.conway_poly(2, 0)
    with pytest.raises(LookupError):
        # GF(2^409) is the largest characteristic-2 field in Frank Luebeck's database
        galois.conway_poly(2, 410)


def test_conway_poly():
    assert galois.conway_poly(2, 8) == galois.Poly.Degrees([8, 4, 3, 2, 0])

    GF3 = galois.GF(3)
    assert galois.conway_poly(3, 8) == galois.Poly.Degrees([8, 5, 4, 2, 1, 0], coeffs=[1, 2, 1, 2, 2, 2], field=GF3)

    GF5 = galois.GF(5)
    assert galois.conway_poly(5, 8) == galois.Poly.Degrees([8, 4, 2, 1, 0], coeffs=[1, 1, 3, 4, 2], field=GF5)


def test_matlab_primitive_poly_exceptions():
    with pytest.raises(TypeError):
        galois.matlab_primitive_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.matlab_primitive_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.matlab_primitive_poly(4, 3)
    with pytest.raises(ValueError):
        galois.matlab_primitive_poly(2, 0)


def test_matlab_primitive_poly():
    """
    Matlab:
        % Note Matlab's ordering is degree-ascending
        gfprimdf(m, p)
    """
    assert galois.matlab_primitive_poly(2, 1).coeffs.tolist()[::-1] == [1, 1]
    assert galois.matlab_primitive_poly(2, 2).coeffs.tolist()[::-1] == [1, 1, 1]
    assert galois.matlab_primitive_poly(2, 3).coeffs.tolist()[::-1] == [1, 1, 0, 1]
    assert galois.matlab_primitive_poly(2, 4).coeffs.tolist()[::-1] == [1, 1, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 5).coeffs.tolist()[::-1] == [1, 0, 1, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 6).coeffs.tolist()[::-1] == [1, 1, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 7).coeffs.tolist()[::-1] == [1, 0, 0, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 8).coeffs.tolist()[::-1] == [1, 0, 1, 1, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 9).coeffs.tolist()[::-1] == [1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 10).coeffs.tolist()[::-1] == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 11).coeffs.tolist()[::-1] == [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 12).coeffs.tolist()[::-1] == [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 13).coeffs.tolist()[::-1] == [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 14).coeffs.tolist()[::-1] == [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 15).coeffs.tolist()[::-1] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(2, 16).coeffs.tolist()[::-1] == [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]

    assert galois.matlab_primitive_poly(3, 1).coeffs.tolist()[::-1] == [1, 1]
    assert galois.matlab_primitive_poly(3, 2).coeffs.tolist()[::-1] == [2, 1, 1]
    assert galois.matlab_primitive_poly(3, 3).coeffs.tolist()[::-1] == [1, 2, 0, 1]
    assert galois.matlab_primitive_poly(3, 4).coeffs.tolist()[::-1] == [2, 1, 0, 0, 1]
    assert galois.matlab_primitive_poly(3, 5).coeffs.tolist()[::-1] == [1, 2, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(3, 6).coeffs.tolist()[::-1] == [2, 1, 0, 0, 0, 0, 1]
    # assert galois.matlab_primitive_poly(3, 7).coeffs.tolist()[::-1] == [1, 0, 2, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(3, 8).coeffs.tolist()[::-1] == [2, 0, 0, 1, 0, 0, 0, 0, 1]

    assert galois.matlab_primitive_poly(5, 1).coeffs.tolist()[::-1] == [2, 1]
    assert galois.matlab_primitive_poly(5, 2).coeffs.tolist()[::-1] == [2, 1, 1]
    assert galois.matlab_primitive_poly(5, 3).coeffs.tolist()[::-1] == [2, 3, 0, 1]
    # assert galois.matlab_primitive_poly(5, 4).coeffs.tolist()[::-1] == [2, 2, 1, 0, 1]
    assert galois.matlab_primitive_poly(5, 5).coeffs.tolist()[::-1] == [2, 4, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(5, 6).coeffs.tolist()[::-1] == [2, 1, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(5, 7).coeffs.tolist()[::-1] == [2, 3, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(5, 8).coeffs.tolist()[::-1] == [3, 2, 1, 0, 0, 0, 0, 0, 1]

    assert galois.matlab_primitive_poly(7, 1).coeffs.tolist()[::-1] == [2, 1]
    assert galois.matlab_primitive_poly(7, 2).coeffs.tolist()[::-1] == [3, 1, 1]
    assert galois.matlab_primitive_poly(7, 3).coeffs.tolist()[::-1] == [2, 3, 0, 1]
    assert galois.matlab_primitive_poly(7, 4).coeffs.tolist()[::-1] == [5, 3, 1, 0, 1]
    assert galois.matlab_primitive_poly(7, 5).coeffs.tolist()[::-1] == [4, 1, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(7, 6).coeffs.tolist()[::-1] == [5, 1, 3, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(7, 7).coeffs.tolist()[::-1] == [2, 6, 0, 0, 0, 0, 0, 1]
    assert galois.matlab_primitive_poly(7, 8).coeffs.tolist()[::-1] == [3, 1, 0, 0, 0, 0, 0, 0, 1]
