"""
A pytest module to test generating primitive polynomials and testing primitivity.

Sage:
    p = 3
    for degree in range(1, 7):
        print(f"PRIMITIVE_POLY_{p}_{degree} = [")
        F = GF(p)["x"]
        for f in F.polynomials(degree):
            # For some reason `is_primitive()` crashes on f(x) = x
            if f.coefficients(sparse=False) == [0, 1]:
                continue
            if f.is_monic() and f.is_primitive():
                c = f.coefficients(sparse=False)[::-1]
                print(f"    galois.Poly({c}, field=GF{p}),")
        print("]\n")

References
----------
* https://baylor-ir.tdl.org/bitstream/handle/2104/8793/GF3%20Polynomials.pdf?sequence=1&isAllowed=y
"""
import pytest

import galois

GF2 = galois.GF(2)
GF3 = galois.GF(3)
GF5 = galois.GF(5)
GF7 = galois.GF(7)

PRIMITIVE_POLYS_2_1 = [
    galois.Poly([1, 1], field=GF2),
]

PRIMITIVE_POLYS_2_2 = [
    galois.Poly([1, 1, 1], field=GF2),
]

PRIMITIVE_POLYS_2_3 = [
    galois.Poly([1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1], field=GF2),
]

PRIMITIVE_POLYS_2_4 = [
    galois.Poly([1, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 1], field=GF2),
]

PRIMITIVE_POLYS_2_5 = [
    galois.Poly([1, 0, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 1], field=GF2),
]

PRIMITIVE_POLYS_2_6 = [
    galois.Poly([1, 0, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 1, 1], field=GF2),
]

PRIMITIVE_POLYS_2_7 = [
    galois.Poly([1, 0, 0, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 0, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 1, 1, 0, 1], field=GF2),
]

PRIMITIVE_POLYS_2_8 = [
    galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 0, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 1, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 1, 0, 1, 0, 1], field=GF2),
]

PRIMITIVE_POLYS_3_1 = [
    galois.Poly([1, 1], field=GF3),
]

PRIMITIVE_POLYS_3_2 = [
    galois.Poly([1, 1, 2], field=GF3),
    galois.Poly([1, 2, 2], field=GF3),
]

PRIMITIVE_POLYS_3_3 = [
    galois.Poly([1, 0, 2, 1], field=GF3),
    galois.Poly([1, 1, 2, 1], field=GF3),
    galois.Poly([1, 2, 0, 1], field=GF3),
    galois.Poly([1, 2, 1, 1], field=GF3),
]

PRIMITIVE_POLYS_3_4 = [
    galois.Poly([1, 0, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 2, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 2, 2], field=GF3),
    galois.Poly([1, 2, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 1, 1, 2], field=GF3),
    galois.Poly([1, 2, 2, 1, 2], field=GF3),
]

PRIMITIVE_POLYS_3_5 = [
    galois.Poly([1, 0, 0, 0, 2, 1], field=GF3),
    galois.Poly([1, 0, 0, 2, 1, 1], field=GF3),
    galois.Poly([1, 0, 1, 0, 1, 1], field=GF3),
    galois.Poly([1, 0, 1, 2, 0, 1], field=GF3),
    galois.Poly([1, 0, 1, 2, 2, 1], field=GF3),
    galois.Poly([1, 0, 2, 1, 0, 1], field=GF3),
    galois.Poly([1, 0, 2, 2, 1, 1], field=GF3),
    galois.Poly([1, 1, 0, 0, 2, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 0, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 0, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 1, 2, 1], field=GF3),
    galois.Poly([1, 1, 1, 2, 1, 1], field=GF3),
    galois.Poly([1, 1, 2, 0, 0, 1], field=GF3),
    galois.Poly([1, 1, 2, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 2, 2, 0, 1], field=GF3),
    galois.Poly([1, 2, 0, 0, 0, 1], field=GF3),
    galois.Poly([1, 2, 0, 0, 1, 1], field=GF3),
    galois.Poly([1, 2, 0, 2, 2, 1], field=GF3),
    galois.Poly([1, 2, 1, 1, 1, 1], field=GF3),
    galois.Poly([1, 2, 2, 0, 2, 1], field=GF3),
    galois.Poly([1, 2, 2, 1, 0, 1], field=GF3),
]

PRIMITIVE_POLYS_3_6 = [
    galois.Poly([1, 0, 0, 0, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 0, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 0, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 2, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 0, 2, 1, 2], field=GF3),
    galois.Poly([1, 0, 1, 0, 2, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 1, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 1, 1, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 2, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 1, 2, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 0, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 0, 1, 2, 2], field=GF3),
    galois.Poly([1, 0, 2, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 0, 1, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 0, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 1, 0, 1, 2, 1, 2], field=GF3),
    galois.Poly([1, 1, 0, 2, 2, 0, 2], field=GF3),
    galois.Poly([1, 1, 1, 0, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 1, 0, 2, 0, 2], field=GF3),
    galois.Poly([1, 1, 1, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 1, 1, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 1, 1, 1, 2, 2, 2], field=GF3),
    galois.Poly([1, 1, 1, 2, 2, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 0, 1, 0, 2], field=GF3),
    galois.Poly([1, 1, 2, 0, 2, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 1, 0, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 2, 1, 2], field=GF3),
    galois.Poly([1, 1, 2, 2, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 0, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 1, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 2, 2, 2], field=GF3),
    galois.Poly([1, 2, 1, 0, 1, 1, 2], field=GF3),
    galois.Poly([1, 2, 1, 0, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 1, 1, 2, 1, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 0, 2, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 2, 1, 2], field=GF3),
    galois.Poly([1, 2, 2, 0, 1, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 0, 2, 1, 2], field=GF3),
    galois.Poly([1, 2, 2, 1, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 2, 0, 2, 2], field=GF3),
    galois.Poly([1, 2, 2, 2, 1, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 2, 2, 2, 2], field=GF3),
]

PRIMITIVE_POLYS_5_1 = [
    galois.Poly([1, 2], field=GF5),
    galois.Poly([1, 3], field=GF5),
]

PRIMITIVE_POLYS_5_2 = [
    galois.Poly([1, 1, 2], field=GF5),
    galois.Poly([1, 2, 3], field=GF5),
    galois.Poly([1, 3, 3], field=GF5),
    galois.Poly([1, 4, 2], field=GF5),
]

PRIMITIVE_POLYS_5_3 = [
    galois.Poly([1, 0, 3, 2], field=GF5),
    galois.Poly([1, 0, 3, 3], field=GF5),
    galois.Poly([1, 0, 4, 2], field=GF5),
    galois.Poly([1, 0, 4, 3], field=GF5),
    galois.Poly([1, 1, 0, 2], field=GF5),
    galois.Poly([1, 1, 1, 3], field=GF5),
    galois.Poly([1, 1, 4, 3], field=GF5),
    galois.Poly([1, 2, 0, 3], field=GF5),
    galois.Poly([1, 2, 1, 3], field=GF5),
    galois.Poly([1, 2, 2, 2], field=GF5),
    galois.Poly([1, 2, 2, 3], field=GF5),
    galois.Poly([1, 2, 4, 2], field=GF5),
    galois.Poly([1, 3, 0, 2], field=GF5),
    galois.Poly([1, 3, 1, 2], field=GF5),
    galois.Poly([1, 3, 2, 2], field=GF5),
    galois.Poly([1, 3, 2, 3], field=GF5),
    galois.Poly([1, 3, 4, 3], field=GF5),
    galois.Poly([1, 4, 0, 3], field=GF5),
    galois.Poly([1, 4, 1, 2], field=GF5),
    galois.Poly([1, 4, 4, 2], field=GF5),
]

PRIMITIVE_POLYS_5_4 = [
    galois.Poly([1, 0, 1, 2, 2], field=GF5),
    galois.Poly([1, 0, 1, 2, 3], field=GF5),
    galois.Poly([1, 0, 1, 3, 2], field=GF5),
    galois.Poly([1, 0, 1, 3, 3], field=GF5),
    galois.Poly([1, 0, 4, 1, 2], field=GF5),
    galois.Poly([1, 0, 4, 1, 3], field=GF5),
    galois.Poly([1, 0, 4, 4, 2], field=GF5),
    galois.Poly([1, 0, 4, 4, 3], field=GF5),
    galois.Poly([1, 1, 0, 1, 3], field=GF5),
    galois.Poly([1, 1, 0, 2, 3], field=GF5),
    galois.Poly([1, 1, 0, 3, 2], field=GF5),
    galois.Poly([1, 1, 0, 4, 2], field=GF5),
    galois.Poly([1, 1, 1, 1, 3], field=GF5),
    galois.Poly([1, 1, 2, 0, 2], field=GF5),
    galois.Poly([1, 1, 2, 1, 2], field=GF5),
    galois.Poly([1, 1, 3, 0, 3], field=GF5),
    galois.Poly([1, 1, 3, 4, 2], field=GF5),
    galois.Poly([1, 1, 4, 4, 3], field=GF5),
    galois.Poly([1, 2, 0, 1, 3], field=GF5),
    galois.Poly([1, 2, 0, 2, 2], field=GF5),
    galois.Poly([1, 2, 0, 3, 3], field=GF5),
    galois.Poly([1, 2, 0, 4, 2], field=GF5),
    galois.Poly([1, 2, 1, 2, 3], field=GF5),
    galois.Poly([1, 2, 2, 0, 3], field=GF5),
    galois.Poly([1, 2, 2, 2, 2], field=GF5),
    galois.Poly([1, 2, 3, 0, 2], field=GF5),
    galois.Poly([1, 2, 3, 3, 2], field=GF5),
    galois.Poly([1, 2, 4, 3, 3], field=GF5),
    galois.Poly([1, 3, 0, 1, 2], field=GF5),
    galois.Poly([1, 3, 0, 2, 3], field=GF5),
    galois.Poly([1, 3, 0, 3, 2], field=GF5),
    galois.Poly([1, 3, 0, 4, 3], field=GF5),
    galois.Poly([1, 3, 1, 3, 3], field=GF5),
    galois.Poly([1, 3, 2, 0, 3], field=GF5),
    galois.Poly([1, 3, 2, 3, 2], field=GF5),
    galois.Poly([1, 3, 3, 0, 2], field=GF5),
    galois.Poly([1, 3, 3, 2, 2], field=GF5),
    galois.Poly([1, 3, 4, 2, 3], field=GF5),
    galois.Poly([1, 4, 0, 1, 2], field=GF5),
    galois.Poly([1, 4, 0, 2, 2], field=GF5),
    galois.Poly([1, 4, 0, 3, 3], field=GF5),
    galois.Poly([1, 4, 0, 4, 3], field=GF5),
    galois.Poly([1, 4, 1, 4, 3], field=GF5),
    galois.Poly([1, 4, 2, 0, 2], field=GF5),
    galois.Poly([1, 4, 2, 4, 2], field=GF5),
    galois.Poly([1, 4, 3, 0, 3], field=GF5),
    galois.Poly([1, 4, 3, 1, 2], field=GF5),
    galois.Poly([1, 4, 4, 1, 3], field=GF5),
]


def test_primitive_poly_exceptions():
    with pytest.raises(TypeError):
        galois.primitive_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.primitive_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.primitive_poly(4, 3)
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 0)
    with pytest.raises(ValueError):
        galois.primitive_poly(2, 3, method="invalid-argument")


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_primitive_poly_min(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert galois.primitive_poly(characteristic, degree) == LUT[0]


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_primitive_poly_max(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert galois.primitive_poly(characteristic, degree, method="max") == LUT[-1]


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_primitive_poly_random(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert galois.primitive_poly(characteristic, degree, method="random") in LUT


def test_primitive_polys_exceptions():
    with pytest.raises(TypeError):
        galois.primitive_polys(2.0, 3)
    with pytest.raises(TypeError):
        galois.primitive_polys(2, 3.0)
    with pytest.raises(ValueError):
        galois.primitive_polys(4, 3)
    with pytest.raises(ValueError):
        galois.primitive_polys(2, 0)


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_primitive_polys(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert galois.primitive_polys(characteristic, degree) == LUT


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
        # GF(2^409) is the largest 2-characteristic field in Frank Luebeck's database
        galois.conway_poly(2, 410)


def test_conway_poly():
    assert galois.conway_poly(2, 8) == galois.Poly.Degrees([8, 4, 3, 2, 0])
    assert galois.conway_poly(3, 8) == galois.Poly.Degrees([8, 5, 4, 2, 1, 0], coeffs=[1, 2, 1, 2, 2, 2], field=GF3)
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
        gfprimdf(m, 2)
        gfprimdf(m, 3)
        gfprimdf(m, 5)
        gfprimdf(m, 7)
    """
    assert galois.matlab_primitive_poly(2, 1) == galois.Poly([1,1], order="asc")
    assert galois.matlab_primitive_poly(2, 2) == galois.Poly([1,1,1], order="asc")
    assert galois.matlab_primitive_poly(2, 3) == galois.Poly([1,1,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 4) == galois.Poly([1,1,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 5) == galois.Poly([1,0,1,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 6) == galois.Poly([1,1,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 7) == galois.Poly([1,0,0,1,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 8) == galois.Poly([1,0,1,1,1,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 9) == galois.Poly([1,0,0,0,1,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 10) == galois.Poly([1,0,0,1,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 11) == galois.Poly([1,0,1,0,0,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 12) == galois.Poly([1,1,0,0,1,0,1,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 13) == galois.Poly([1,1,0,1,1,0,0,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 14) == galois.Poly([1,1,0,0,0,0,1,0,0,0,1,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 15) == galois.Poly([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1], order="asc")
    assert galois.matlab_primitive_poly(2, 16) == galois.Poly([1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1], order="asc")

    assert galois.matlab_primitive_poly(3, 1) == galois.Poly([1,1], field=GF3, order="asc")
    assert galois.matlab_primitive_poly(3, 2) == galois.Poly([2,1,1], field=GF3, order="asc")
    assert galois.matlab_primitive_poly(3, 3) == galois.Poly([1,2,0,1], field=GF3, order="asc")
    assert galois.matlab_primitive_poly(3, 4) == galois.Poly([2,1,0,0,1], field=GF3, order="asc")
    assert galois.matlab_primitive_poly(3, 5) == galois.Poly([1,2,0,0,0,1], field=GF3, order="asc")
    assert galois.matlab_primitive_poly(3, 6) == galois.Poly([2,1,0,0,0,0,1], field=GF3, order="asc")
    # assert galois.matlab_primitive_poly(3, 7) == galois.Poly([1,0,2,0,0,0,0,1], field=GF3, order="asc")
    assert galois.matlab_primitive_poly(3, 8) == galois.Poly([2,0,0,1,0,0,0,0,1], field=GF3, order="asc")

    assert galois.matlab_primitive_poly(5, 1) == galois.Poly([2,1], field=GF5, order="asc")
    assert galois.matlab_primitive_poly(5, 2) == galois.Poly([2,1,1], field=GF5, order="asc")
    assert galois.matlab_primitive_poly(5, 3) == galois.Poly([2,3,0,1], field=GF5, order="asc")
    # assert galois.matlab_primitive_poly(5, 4) == galois.Poly([2,2,1,0,1], field=GF5, order="asc")
    assert galois.matlab_primitive_poly(5, 5) == galois.Poly([2,4,0,0,0,1], field=GF5, order="asc")
    assert galois.matlab_primitive_poly(5, 6) == galois.Poly([2,1,0,0,0,0,1], field=GF5, order="asc")
    assert galois.matlab_primitive_poly(5, 7) == galois.Poly([2,3,0,0,0,0,0,1], field=GF5, order="asc")
    assert galois.matlab_primitive_poly(5, 8) == galois.Poly([3,2,1,0,0,0,0,0,1], field=GF5, order="asc")

    assert galois.matlab_primitive_poly(7, 1) == galois.Poly([2,1], field=GF7, order="asc")
    assert galois.matlab_primitive_poly(7, 2) == galois.Poly([3,1,1], field=GF7, order="asc")
    assert galois.matlab_primitive_poly(7, 3) == galois.Poly([2,3,0,1], field=GF7, order="asc")
    assert galois.matlab_primitive_poly(7, 4) == galois.Poly([5,3,1,0,1], field=GF7, order="asc")
    assert galois.matlab_primitive_poly(7, 5) == galois.Poly([4,1,0,0,0,1], field=GF7, order="asc")
    assert galois.matlab_primitive_poly(7, 6) == galois.Poly([5,1,3,0,0,0,1], field=GF7, order="asc")
    assert galois.matlab_primitive_poly(7, 7) == galois.Poly([2,6,0,0,0,0,0,1], field=GF7, order="asc")
    assert galois.matlab_primitive_poly(7, 8) == galois.Poly([3,1,0,0,0,0,0,0,1], field=GF7, order="asc")


def test_is_primitive_exceptions():
    with pytest.raises(TypeError):
        galois.is_primitive([1, 0, 1, 1])
    with pytest.raises(ValueError):
        galois.is_primitive(galois.Poly([1]))
    with pytest.raises(ValueError):
        galois.is_primitive(galois.Poly([1], field=galois.GF(2**2)))


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_is_primitive(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert all(galois.is_primitive(f) for f in LUT)


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,2), (3,3), (3,4), (3,5), (3,6), (5,2), (5,3), (5,4)])
def test_is_not_primitive(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    while True:
        f = galois.Poly.Random(degree, field=galois.GF(characteristic))
        f /= f.coeffs[0]  # Make monic
        if f not in LUT:
            break
    assert not galois.is_primitive(f)
