"""
A pytest module to test computing derivatives of polynomials over Galois fields.

Sage:
    def to_str(poly):
        c = poly.coefficients(sparse=False)[::-1]
        return f"galois.Poly({c}, field=GF{characteristic}_{degree})"

    N = 20
    for (characteristic, degree) in [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)]:
        print(f"POLY_DERIVATIVES_{characteristic}_{degree} = [")
        R = GF(characteristic**degree, repr="int")["x"]
        for _ in range(N):
            poly = R.random_element(randint(0, 10))
            k = randint(1, characteristic-1)
            d_poly = poly.derivative(k)
            print(f"    ({to_str(poly)}, {k}, {to_str(d_poly)}),")
        print("]\n")
"""
import pytest

import galois

GF2_1 = galois.GF(2**1)
GF2_8 = galois.GF(2**8)
GF3_1 = galois.GF(3**1)
GF3_5 = galois.GF(3**5)
GF5_1 = galois.GF(5**1)
GF5_4 = galois.GF(5**4)

POLY_DERIVATIVES_2_1 = [
    (galois.Poly([1, 0, 1, 0, 1, 0, 1], field=GF2_1), 1, galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 1, 0, 0, 0], field=GF2_1), 1, galois.Poly([1, 0, 0], field=GF2_1)),
    (galois.Poly([1], field=GF2_1), 1, galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 0, 1, 0, 1, 0, 0], field=GF2_1), 1, galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 1, 1, 0, 1, 1, 1, 1, 1, 1], field=GF2_1), 1, galois.Poly([1, 0, 1, 0, 1, 0, 1, 0, 1], field=GF2_1)),
    (galois.Poly([1, 0, 0, 0, 1, 0, 1, 0, 1, 0], field=GF2_1), 1, galois.Poly([1, 0, 0, 0, 1, 0, 1, 0, 1], field=GF2_1)),
    (galois.Poly([1, 0, 0, 1, 0, 1, 1], field=GF2_1), 1, galois.Poly([1, 0, 1], field=GF2_1)),
    (galois.Poly([1, 1, 0, 1, 0, 1, 1, 1, 1, 1], field=GF2_1), 1, galois.Poly([1, 0, 0, 0, 0, 0, 1, 0, 1], field=GF2_1)),
    (galois.Poly([1, 0, 0, 0, 1, 0, 1], field=GF2_1), 1, galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 0, 0, 0, 0, 0, 0], field=GF2_1), 1, galois.Poly([], field=GF2_1)),
    (galois.Poly([1], field=GF2_1), 1, galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 1, 1, 0, 0, 0, 0, 0], field=GF2_1), 1, galois.Poly([1, 0, 1, 0, 0, 0, 0], field=GF2_1)),
    (galois.Poly([1, 0, 0, 0, 1, 1, 1, 1, 0], field=GF2_1), 1, galois.Poly([1, 0, 1], field=GF2_1)),
    (galois.Poly([1, 1], field=GF2_1), 1, galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 1, 0, 0, 1, 0, 1, 0, 0, 0], field=GF2_1), 1, galois.Poly([1, 0, 0, 0, 1, 0, 1, 0, 0], field=GF2_1)),
    (galois.Poly([1, 1, 1, 1, 0], field=GF2_1), 1, galois.Poly([1, 0, 1], field=GF2_1)),
    (galois.Poly([1, 0, 0, 1, 0, 1, 1, 1], field=GF2_1), 1, galois.Poly([1, 0, 0, 0, 0, 0, 1], field=GF2_1)),
    (galois.Poly([1, 0, 0, 0, 1], field=GF2_1), 1, galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 1, 1, 0, 1, 1, 0], field=GF2_1), 1, galois.Poly([1, 0, 0, 0, 1], field=GF2_1)),
    (galois.Poly([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], field=GF2_1), 1, galois.Poly([1, 0, 1, 0, 0, 0, 0], field=GF2_1)),
]

POLY_DERIVATIVES_2_8 = [
    (galois.Poly([45, 40, 124, 59], field=GF2_8), 1, galois.Poly([45, 0, 124], field=GF2_8)),
    (galois.Poly([32, 58, 128, 194, 209], field=GF2_8), 1, galois.Poly([58, 0, 194], field=GF2_8)),
    (galois.Poly([43], field=GF2_8), 1, galois.Poly([], field=GF2_8)),
    (galois.Poly([105, 214, 204, 34, 74, 43], field=GF2_8), 1, galois.Poly([105, 0, 204, 0, 74], field=GF2_8)),
    (galois.Poly([177, 192, 211, 132], field=GF2_8), 1, galois.Poly([177, 0, 211], field=GF2_8)),
    (galois.Poly([202, 178, 42, 56, 28, 183, 101, 121, 98, 22, 207], field=GF2_8), 1, galois.Poly([178, 0, 56, 0, 183, 0, 121, 0, 22], field=GF2_8)),
    (galois.Poly([46, 167, 8, 43, 162, 160, 0, 97, 77, 155, 47], field=GF2_8), 1, galois.Poly([167, 0, 43, 0, 160, 0, 97, 0, 155], field=GF2_8)),
    (galois.Poly([106, 84, 66], field=GF2_8), 1, galois.Poly([84], field=GF2_8)),
    (galois.Poly([32, 11, 179, 104], field=GF2_8), 1, galois.Poly([32, 0, 179], field=GF2_8)),
    (galois.Poly([192, 102, 49], field=GF2_8), 1, galois.Poly([102], field=GF2_8)),
    (galois.Poly([196, 72, 215, 81, 152, 31, 124, 37], field=GF2_8), 1, galois.Poly([196, 0, 215, 0, 152, 0, 124], field=GF2_8)),
    (galois.Poly([160, 210, 227, 238], field=GF2_8), 1, galois.Poly([160, 0, 227], field=GF2_8)),
    (galois.Poly([4, 143, 236], field=GF2_8), 1, galois.Poly([143], field=GF2_8)),
    (galois.Poly([240, 33, 228, 35, 247, 174, 90, 115, 229, 192], field=GF2_8), 1, galois.Poly([240, 0, 228, 0, 247, 0, 90, 0, 229], field=GF2_8)),
    (galois.Poly([18, 44, 14, 196, 186, 211, 143, 126, 36, 215], field=GF2_8), 1, galois.Poly([18, 0, 14, 0, 186, 0, 143, 0, 36], field=GF2_8)),
    (galois.Poly([193, 147, 50, 189, 238], field=GF2_8), 1, galois.Poly([147, 0, 189], field=GF2_8)),
    (galois.Poly([123, 177, 6, 134, 166, 85, 84, 105, 210, 142], field=GF2_8), 1, galois.Poly([123, 0, 6, 0, 166, 0, 84, 0, 210], field=GF2_8)),
    (galois.Poly([176], field=GF2_8), 1, galois.Poly([], field=GF2_8)),
    (galois.Poly([121, 65, 171, 160, 104], field=GF2_8), 1, galois.Poly([65, 0, 160], field=GF2_8)),
    (galois.Poly([88, 24, 20, 49, 123], field=GF2_8), 1, galois.Poly([24, 0, 49], field=GF2_8)),
]

POLY_DERIVATIVES_3_1 = [
    (galois.Poly([1, 2, 0], field=GF3_1), 2, galois.Poly([2], field=GF3_1)),
    (galois.Poly([1, 1, 0], field=GF3_1), 1, galois.Poly([2, 1], field=GF3_1)),
    (galois.Poly([2, 0, 1, 2, 1, 2, 1], field=GF3_1), 1, galois.Poly([1, 0, 2, 2], field=GF3_1)),
    (galois.Poly([1, 0, 2], field=GF3_1), 1, galois.Poly([2, 0], field=GF3_1)),
    (galois.Poly([1, 1, 1, 2, 1, 0, 2, 0], field=GF3_1), 1, galois.Poly([1, 0, 2, 2, 0, 0, 2], field=GF3_1)),
    (galois.Poly([2], field=GF3_1), 2, galois.Poly([], field=GF3_1)),
    (galois.Poly([1, 0, 0], field=GF3_1), 2, galois.Poly([2], field=GF3_1)),
    (galois.Poly([2, 0], field=GF3_1), 2, galois.Poly([], field=GF3_1)),
    (galois.Poly([1, 1], field=GF3_1), 2, galois.Poly([], field=GF3_1)),
    (galois.Poly([1, 0, 1, 0, 2], field=GF3_1), 1, galois.Poly([1, 0, 2, 0], field=GF3_1)),
    (galois.Poly([1, 1], field=GF3_1), 2, galois.Poly([], field=GF3_1)),
    (galois.Poly([2, 2, 1, 2, 0], field=GF3_1), 2, galois.Poly([2], field=GF3_1)),
    (galois.Poly([2, 1, 0, 1, 0, 0, 0, 2, 1, 2, 0], field=GF3_1), 2, galois.Poly([2], field=GF3_1)),
    (galois.Poly([1, 1, 2, 0, 0, 1, 2, 1, 0], field=GF3_1), 2, galois.Poly([2, 0, 0, 0, 0, 0, 1], field=GF3_1)),
    (galois.Poly([1], field=GF3_1), 1, galois.Poly([], field=GF3_1)),
    (galois.Poly([2, 0, 1, 0], field=GF3_1), 1, galois.Poly([1], field=GF3_1)),
    (galois.Poly([1, 2, 0, 2, 1, 0, 1, 0], field=GF3_1), 2, galois.Poly([], field=GF3_1)),
    (galois.Poly([1, 1], field=GF3_1), 1, galois.Poly([1], field=GF3_1)),
    (galois.Poly([2, 0, 1, 2, 1, 2, 1, 1], field=GF3_1), 1, galois.Poly([2, 0, 2, 2, 0, 1, 1], field=GF3_1)),
    (galois.Poly([2, 0, 2], field=GF3_1), 1, galois.Poly([1, 0], field=GF3_1)),
]

POLY_DERIVATIVES_3_5 = [
    (galois.Poly([228, 95, 56, 44, 14, 21], field=GF3_5), 2, galois.Poly([132, 0, 0, 76], field=GF3_5)),
    (galois.Poly([73, 166, 40, 93, 221, 130, 178, 225, 209, 57, 117], field=GF3_5), 1, galois.Poly([73, 0, 80, 93, 0, 233, 178, 0, 145, 57], field=GF3_5)),
    (galois.Poly([225, 190, 0, 39, 14], field=GF3_5), 1, galois.Poly([225, 0, 0, 39], field=GF3_5)),
    (galois.Poly([94, 114, 25, 28, 60, 87, 239, 42, 171, 227], field=GF3_5), 1, galois.Poly([219, 25, 0, 30, 87, 0, 75, 171], field=GF3_5)),
    (galois.Poly([26, 37, 214], field=GF3_5), 2, galois.Poly([13], field=GF3_5)),
    (galois.Poly([47, 233, 236], field=GF3_5), 2, galois.Poly([64], field=GF3_5)),
    (galois.Poly([187], field=GF3_5), 2, galois.Poly([], field=GF3_5)),
    (galois.Poly([231, 54, 209, 12, 205, 139], field=GF3_5), 1, galois.Poly([129, 54, 0, 24, 205], field=GF3_5)),
    (galois.Poly([61, 143, 200, 157, 15, 184, 185], field=GF3_5), 2, galois.Poly([193, 0, 0, 21], field=GF3_5)),
    (galois.Poly([51, 85, 25], field=GF3_5), 1, galois.Poly([66, 85], field=GF3_5)),
    (galois.Poly([159, 3, 150, 136, 200], field=GF3_5), 1, galois.Poly([159, 0, 210, 136], field=GF3_5)),
    (galois.Poly([41, 242, 96, 182], field=GF3_5), 1, galois.Poly([121, 96], field=GF3_5)),
    (galois.Poly([25, 210, 111, 136], field=GF3_5), 1, galois.Poly([150, 111], field=GF3_5)),
    (galois.Poly([140, 183, 151, 25, 169, 204, 148, 203, 148, 67], field=GF3_5), 1, galois.Poly([96, 151, 0, 86, 204, 0, 160, 148], field=GF3_5)),
    (galois.Poly([98, 177, 73, 13, 155, 145, 128, 216, 215, 238], field=GF3_5), 1, galois.Poly([102, 73, 0, 199, 145, 0, 108, 215], field=GF3_5)),
    (galois.Poly([228, 78, 86, 144], field=GF3_5), 2, galois.Poly([39], field=GF3_5)),
    (galois.Poly([104, 175, 49], field=GF3_5), 1, galois.Poly([178, 175], field=GF3_5)),
    (galois.Poly([148, 140, 174, 191, 66, 106, 178, 13], field=GF3_5), 1, galois.Poly([148, 0, 105, 191, 0, 176, 178], field=GF3_5)),
    (galois.Poly([95, 133, 58, 87, 98, 69, 40, 208, 151, 64, 185], field=GF3_5), 1, galois.Poly([95, 0, 35, 87, 0, 48, 40, 0, 212, 64], field=GF3_5)),
    (galois.Poly([29, 145, 80, 99, 85, 18], field=GF3_5), 1, galois.Poly([55, 145, 0, 171, 85], field=GF3_5)),
]

POLY_DERIVATIVES_5_1 = [
    (galois.Poly([2, 4], field=GF5_1), 2, galois.Poly([], field=GF5_1)),
    (galois.Poly([1, 1, 3, 4, 0, 3, 3, 1, 0, 1], field=GF5_1), 3, galois.Poly([4, 1, 0, 0, 0, 2, 3], field=GF5_1)),
    (galois.Poly([3, 0, 0, 2, 4, 4], field=GF5_1), 4, galois.Poly([], field=GF5_1)),
    (galois.Poly([4, 0], field=GF5_1), 3, galois.Poly([], field=GF5_1)),
    (galois.Poly([1, 4, 4, 1], field=GF5_1), 2, galois.Poly([1, 3], field=GF5_1)),
    (galois.Poly([2, 4, 1, 4, 1], field=GF5_1), 4, galois.Poly([3], field=GF5_1)),
    (galois.Poly([4, 1], field=GF5_1), 2, galois.Poly([], field=GF5_1)),
    (galois.Poly([3, 1, 0, 2, 1, 3, 1, 3, 1, 2], field=GF5_1), 3, galois.Poly([2, 1, 0, 0, 0, 2, 1], field=GF5_1)),
    (galois.Poly([1, 3, 0, 0, 4, 4], field=GF5_1), 2, galois.Poly([1, 0, 0], field=GF5_1)),
    (galois.Poly([4, 4, 1, 2, 4, 3, 0, 0, 0, 2], field=GF5_1), 3, galois.Poly([1, 4, 0, 0, 0, 2, 0], field=GF5_1)),
    (galois.Poly([2, 0], field=GF5_1), 4, galois.Poly([], field=GF5_1)),
    (galois.Poly([4, 3, 0, 4, 2, 3, 0, 3, 2, 3, 0], field=GF5_1), 3, galois.Poly([2, 0, 0, 0, 0, 0, 3], field=GF5_1)),
    (galois.Poly([3, 1, 3, 4, 0, 3], field=GF5_1), 4, galois.Poly([4], field=GF5_1)),
    (galois.Poly([3, 2, 2, 3, 3, 4, 2, 2, 4, 4, 2], field=GF5_1), 4, galois.Poly([3, 0, 0, 0, 0, 3], field=GF5_1)),
    (galois.Poly([3, 4, 0, 1, 0, 3, 4, 1], field=GF5_1), 2, galois.Poly([1, 0, 0, 2, 0, 1], field=GF5_1)),
    (galois.Poly([2, 4, 3, 3, 1, 2, 4, 2], field=GF5_1), 2, galois.Poly([4, 0, 0, 1, 1, 4], field=GF5_1)),
    (galois.Poly([1, 2, 0, 2, 2, 2, 4, 4, 0], field=GF5_1), 1, galois.Poly([3, 4, 0, 0, 3, 1, 3, 4], field=GF5_1)),
    (galois.Poly([2, 2, 4], field=GF5_1), 4, galois.Poly([], field=GF5_1)),
    (galois.Poly([2, 3, 0, 1], field=GF5_1), 1, galois.Poly([1, 1, 0], field=GF5_1)),
    (galois.Poly([1, 2, 2, 0], field=GF5_1), 4, galois.Poly([], field=GF5_1)),
]

POLY_DERIVATIVES_5_4 = [
    (galois.Poly([451, 205, 288, 418, 23, 547, 585, 265], field=GF5_4), 4, galois.Poly([362], field=GF5_4)),
    (galois.Poly([53, 323, 293, 96, 448, 233, 420, 425, 530], field=GF5_4), 4, galois.Poly([332], field=GF5_4)),
    (galois.Poly([194, 508, 239, 605], field=GF5_4), 1, galois.Poly([422, 386, 239], field=GF5_4)),
    (galois.Poly([601, 407, 262, 270, 140, 240, 552, 602, 426, 189, 72], field=GF5_4), 2, galois.Poly([189, 262, 515, 0, 0, 479, 602, 227], field=GF5_4)),
    (galois.Poly([206, 238, 423, 568, 185, 292, 145], field=GF5_4), 3, galois.Poly([357, 568], field=GF5_4)),
    (galois.Poly([335, 434, 226, 5, 40, 524, 56], field=GF5_4), 3, galois.Poly([529, 5], field=GF5_4)),
    (galois.Poly([483, 611], field=GF5_4), 2, galois.Poly([], field=GF5_4)),
    (galois.Poly([419, 529, 473, 496, 495, 112], field=GF5_4), 2, galois.Poly([428, 473, 217], field=GF5_4)),
    (galois.Poly([505, 609, 44, 393], field=GF5_4), 4, galois.Poly([], field=GF5_4)),
    (galois.Poly([285, 495, 436, 617], field=GF5_4), 3, galois.Poly([285], field=GF5_4)),
    (galois.Poly([395, 433, 132, 612, 92, 506], field=GF5_4), 2, galois.Poly([236, 132, 474], field=GF5_4)),
    (galois.Poly([268, 481, 426, 424, 415, 41, 71, 513], field=GF5_4), 2, galois.Poly([506, 0, 0, 193, 415, 57], field=GF5_4)),
    (galois.Poly([348, 281], field=GF5_4), 3, galois.Poly([], field=GF5_4)),
    (galois.Poly([378, 365, 113, 279, 148, 281, 298, 263, 571, 509, 40], field=GF5_4), 4, galois.Poly([410, 0, 0, 0, 0, 482], field=GF5_4)),
    (galois.Poly([105, 477, 595, 24, 34, 93, 266, 156, 379, 551, 131], field=GF5_4), 1, galois.Poly([278, 360, 18, 34, 0, 389, 468, 128, 551], field=GF5_4)),
    (galois.Poly([353, 492, 303, 427, 199, 371], field=GF5_4), 3, galois.Poly([288, 303], field=GF5_4)),
    (galois.Poly([95, 374], field=GF5_4), 2, galois.Poly([], field=GF5_4)),
    (galois.Poly([384], field=GF5_4), 1, galois.Poly([], field=GF5_4)),
    (galois.Poly([326, 528, 618], field=GF5_4), 2, galois.Poly([527], field=GF5_4)),
    (galois.Poly([100], field=GF5_4), 2, galois.Poly([], field=GF5_4)),
]


def test_derivative_exceptions():
    p = galois.Poly.Random(5)

    with pytest.raises(TypeError):
        p.derivative(1.0)
    with pytest.raises(ValueError):
        p.derivative(0)
    with pytest.raises(ValueError):
        p.derivative(-1)


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)])
def test_derivative(characteristic, degree):
    LUT = eval(f"POLY_DERIVATIVES_{characteristic}_{degree}")
    for item in LUT:
        poly, k, derivative = item
        assert poly.derivative(k) == derivative
