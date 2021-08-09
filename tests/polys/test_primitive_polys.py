"""
A pytest module to test generating primitive polynomials and testing primitivity.

Sage:
    PARAMS = [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)]
    for p, m in PARAMS:
        print(f"PRIMITIVE_POLYS_{p}_{m} = [")
        R = GF(p)["x"]
        for f in R.polynomials(m):
            # For some reason `is_primitive()` crashes on f(x) = x
            if f.coefficients(sparse=False) == [0, 1]:
                continue
            if f.is_monic() and f.is_primitive():
                print(f"    {f.coefficients(sparse=False)[::-1]},")
        print("]\n")

References
----------
* https://baylor-ir.tdl.org/bitstream/handle/2104/8793/GF3%20Polynomials.pdf?sequence=1&isAllowed=y
"""
import pytest

import galois

PARAMS = [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)]

# LUT items are poly coefficients in degree-descending order

PRIMITIVE_POLYS_2_1 = [
    [1, 1],
]

PRIMITIVE_POLYS_2_2 = [
    [1, 1, 1],
]

PRIMITIVE_POLYS_2_3 = [
    [1, 0, 1, 1],
    [1, 1, 0, 1],
]

PRIMITIVE_POLYS_2_4 = [
    [1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1],
]

PRIMITIVE_POLYS_2_5 = [
    [1, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1],
]

PRIMITIVE_POLYS_2_6 = [
    [1, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 1, 1],
]

PRIMITIVE_POLYS_2_7 = [
    [1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
]

PRIMITIVE_POLYS_2_8 = [
    [1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 0, 1],
]

PRIMITIVE_POLYS_3_1 = [
    [1, 1],
]

PRIMITIVE_POLYS_3_2 = [
    [1, 1, 2],
    [1, 2, 2],
]

PRIMITIVE_POLYS_3_3 = [
    [1, 0, 2, 1],
    [1, 1, 2, 1],
    [1, 2, 0, 1],
    [1, 2, 1, 1],
]

PRIMITIVE_POLYS_3_4 = [
    [1, 0, 0, 1, 2],
    [1, 0, 0, 2, 2],
    [1, 1, 0, 0, 2],
    [1, 1, 1, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 2, 0, 0, 2],
    [1, 2, 1, 1, 2],
    [1, 2, 2, 1, 2],
]

PRIMITIVE_POLYS_3_5 = [
    [1, 0, 0, 0, 2, 1],
    [1, 0, 0, 2, 1, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 2, 0, 1],
    [1, 0, 1, 2, 2, 1],
    [1, 0, 2, 1, 0, 1],
    [1, 0, 2, 2, 1, 1],
    [1, 1, 0, 0, 2, 1],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 2, 1],
    [1, 1, 1, 2, 1, 1],
    [1, 1, 2, 0, 0, 1],
    [1, 1, 2, 1, 1, 1],
    [1, 1, 2, 2, 0, 1],
    [1, 2, 0, 0, 0, 1],
    [1, 2, 0, 0, 1, 1],
    [1, 2, 0, 2, 2, 1],
    [1, 2, 1, 1, 1, 1],
    [1, 2, 2, 0, 2, 1],
    [1, 2, 2, 1, 0, 1],
]

PRIMITIVE_POLYS_3_6 = [
    [1, 0, 0, 0, 0, 1, 2],
    [1, 0, 0, 0, 0, 2, 2],
    [1, 0, 0, 1, 0, 1, 2],
    [1, 0, 0, 2, 0, 2, 2],
    [1, 0, 1, 0, 2, 1, 2],
    [1, 0, 1, 0, 2, 2, 2],
    [1, 0, 1, 1, 0, 2, 2],
    [1, 0, 1, 1, 1, 2, 2],
    [1, 0, 1, 2, 0, 1, 2],
    [1, 0, 1, 2, 1, 1, 2],
    [1, 0, 2, 0, 1, 1, 2],
    [1, 0, 2, 0, 1, 2, 2],
    [1, 0, 2, 1, 1, 1, 2],
    [1, 0, 2, 2, 1, 2, 2],
    [1, 1, 0, 0, 0, 0, 2],
    [1, 1, 0, 1, 0, 0, 2],
    [1, 1, 0, 1, 1, 1, 2],
    [1, 1, 0, 1, 2, 1, 2],
    [1, 1, 0, 2, 2, 0, 2],
    [1, 1, 1, 0, 1, 2, 2],
    [1, 1, 1, 0, 2, 0, 2],
    [1, 1, 1, 1, 0, 1, 2],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2],
    [1, 1, 2, 0, 1, 0, 2],
    [1, 1, 2, 0, 2, 2, 2],
    [1, 1, 2, 1, 0, 1, 2],
    [1, 1, 2, 1, 1, 0, 2],
    [1, 1, 2, 1, 2, 1, 2],
    [1, 1, 2, 2, 2, 0, 2],
    [1, 2, 0, 0, 0, 0, 2],
    [1, 2, 0, 1, 2, 0, 2],
    [1, 2, 0, 2, 0, 0, 2],
    [1, 2, 0, 2, 1, 2, 2],
    [1, 2, 0, 2, 2, 2, 2],
    [1, 2, 1, 0, 1, 1, 2],
    [1, 2, 1, 0, 2, 0, 2],
    [1, 2, 1, 1, 2, 1, 2],
    [1, 2, 1, 2, 0, 2, 2],
    [1, 2, 1, 2, 1, 2, 2],
    [1, 2, 1, 2, 2, 1, 2],
    [1, 2, 2, 0, 1, 0, 2],
    [1, 2, 2, 0, 2, 1, 2],
    [1, 2, 2, 1, 2, 0, 2],
    [1, 2, 2, 2, 0, 2, 2],
    [1, 2, 2, 2, 1, 0, 2],
    [1, 2, 2, 2, 2, 2, 2],
]

PRIMITIVE_POLYS_5_1 = [
    [1, 2],
    [1, 3],
]

PRIMITIVE_POLYS_5_2 = [
    [1, 1, 2],
    [1, 2, 3],
    [1, 3, 3],
    [1, 4, 2],
]

PRIMITIVE_POLYS_5_3 = [
    [1, 0, 3, 2],
    [1, 0, 3, 3],
    [1, 0, 4, 2],
    [1, 0, 4, 3],
    [1, 1, 0, 2],
    [1, 1, 1, 3],
    [1, 1, 4, 3],
    [1, 2, 0, 3],
    [1, 2, 1, 3],
    [1, 2, 2, 2],
    [1, 2, 2, 3],
    [1, 2, 4, 2],
    [1, 3, 0, 2],
    [1, 3, 1, 2],
    [1, 3, 2, 2],
    [1, 3, 2, 3],
    [1, 3, 4, 3],
    [1, 4, 0, 3],
    [1, 4, 1, 2],
    [1, 4, 4, 2],
]

PRIMITIVE_POLYS_5_4 = [
    [1, 0, 1, 2, 2],
    [1, 0, 1, 2, 3],
    [1, 0, 1, 3, 2],
    [1, 0, 1, 3, 3],
    [1, 0, 4, 1, 2],
    [1, 0, 4, 1, 3],
    [1, 0, 4, 4, 2],
    [1, 0, 4, 4, 3],
    [1, 1, 0, 1, 3],
    [1, 1, 0, 2, 3],
    [1, 1, 0, 3, 2],
    [1, 1, 0, 4, 2],
    [1, 1, 1, 1, 3],
    [1, 1, 2, 0, 2],
    [1, 1, 2, 1, 2],
    [1, 1, 3, 0, 3],
    [1, 1, 3, 4, 2],
    [1, 1, 4, 4, 3],
    [1, 2, 0, 1, 3],
    [1, 2, 0, 2, 2],
    [1, 2, 0, 3, 3],
    [1, 2, 0, 4, 2],
    [1, 2, 1, 2, 3],
    [1, 2, 2, 0, 3],
    [1, 2, 2, 2, 2],
    [1, 2, 3, 0, 2],
    [1, 2, 3, 3, 2],
    [1, 2, 4, 3, 3],
    [1, 3, 0, 1, 2],
    [1, 3, 0, 2, 3],
    [1, 3, 0, 3, 2],
    [1, 3, 0, 4, 3],
    [1, 3, 1, 3, 3],
    [1, 3, 2, 0, 3],
    [1, 3, 2, 3, 2],
    [1, 3, 3, 0, 2],
    [1, 3, 3, 2, 2],
    [1, 3, 4, 2, 3],
    [1, 4, 0, 1, 2],
    [1, 4, 0, 2, 2],
    [1, 4, 0, 3, 3],
    [1, 4, 0, 4, 3],
    [1, 4, 1, 4, 3],
    [1, 4, 2, 0, 2],
    [1, 4, 2, 4, 2],
    [1, 4, 3, 0, 3],
    [1, 4, 3, 1, 2],
    [1, 4, 4, 1, 3],
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


@pytest.mark.parametrize("characteristic,degree", PARAMS)
def test_primitive_poly_min(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert galois.primitive_poly(characteristic, degree).coeffs.tolist() == LUT[0]


@pytest.mark.parametrize("characteristic,degree", PARAMS)
def test_primitive_poly_max(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert galois.primitive_poly(characteristic, degree, method="max").coeffs.tolist() == LUT[-1]


@pytest.mark.parametrize("characteristic,degree", PARAMS)
def test_primitive_poly_random(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert galois.primitive_poly(characteristic, degree, method="random").coeffs.tolist() in LUT


def test_primitive_polys_exceptions():
    with pytest.raises(TypeError):
        galois.primitive_polys(2.0, 3)
    with pytest.raises(TypeError):
        galois.primitive_polys(2, 3.0)
    with pytest.raises(ValueError):
        galois.primitive_polys(4, 3)
    with pytest.raises(ValueError):
        galois.primitive_polys(2, 0)


@pytest.mark.parametrize("characteristic,degree", PARAMS)
def test_primitive_polys(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert [f.coeffs.tolist() for f in galois.primitive_polys(characteristic, degree)] == LUT


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


def test_is_primitive_exceptions():
    with pytest.raises(TypeError):
        galois.is_primitive([1, 0, 1, 1])
    with pytest.raises(ValueError):
        galois.is_primitive(galois.Poly([1]))
    with pytest.raises(ValueError):
        galois.is_primitive(galois.Poly([1], field=galois.GF(2**2)))


@pytest.mark.parametrize("characteristic,degree", PARAMS)
def test_is_primitive(characteristic, degree):
    GF = galois.GF(characteristic)
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    assert all(galois.is_primitive(galois.Poly(f, field=GF)) for f in LUT)


@pytest.mark.parametrize("characteristic,degree", PARAMS)
def test_is_not_primitive(characteristic, degree):
    LUT = eval(f"PRIMITIVE_POLYS_{characteristic}_{degree}")
    while True:
        f = galois.Poly.Random(degree, field=galois.GF(characteristic))
        f /= f.coeffs[0]  # Make monic
        if f.coeffs.tolist() not in LUT:
            break
    assert not galois.is_primitive(f)
