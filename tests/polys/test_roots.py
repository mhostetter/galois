"""
A pytest module to test finding the roots of polynomials over Galois fields.

Sage:
    def to_str(poly):
        c = poly.coefficients(sparse=False)[::-1]
        return f"galois.Poly({c}, field=GF{characteristic}_{degree})"

    N = 20
    for (characteristic, degree) in [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)]:
        print(f"POLY_ROOTS_{characteristic}_{degree} = [")
        R = GF(characteristic**degree, repr="int")["x"]
        for _ in range(N):
            poly = R.random_element(randint(0, 10))
            roots = poly.roots()
            if len(roots) == 0:
                r, m = [], []
            else:
                r, m = zip(*roots)
            print(f"    ({to_str(poly)}, {list(r)}, {list(m)}),")
        print("]\n")
"""
import pytest
import numpy as np

import galois

GF2_1 = galois.GF(2**1)
GF2_8 = galois.GF(2**8)
GF3_1 = galois.GF(3**1)
GF3_5 = galois.GF(3**5)
GF5_1 = galois.GF(5**1)
GF5_4 = galois.GF(5**4)

POLY_ROOTS_2_1 = [
    (galois.Poly([1, 0, 0, 1, 0, 0, 1, 1], field=GF2_1), [1], [2]),
    (galois.Poly([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], field=GF2_1), [1], [1]),
    (galois.Poly([1, 0, 0, 1, 0, 0, 1, 0], field=GF2_1), [0], [1]),
    (galois.Poly([1, 1, 0, 0], field=GF2_1), [1, 0], [1, 2]),
    (galois.Poly([1, 0, 1, 0, 0, 0, 1], field=GF2_1), [], []),
    (galois.Poly([1, 0], field=GF2_1), [0], [1]),
    (galois.Poly([1, 0, 1, 1, 1, 1], field=GF2_1), [], []),
    (galois.Poly([1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1], field=GF2_1), [1], [1]),
    (galois.Poly([1, 1, 1, 1], field=GF2_1), [1], [3]),
    (galois.Poly([1, 1, 0, 0, 0, 1, 1, 0, 0], field=GF2_1), [0, 1], [2, 2]),
    (galois.Poly([1, 0, 1, 0, 0], field=GF2_1), [0, 1], [2, 2]),
    (galois.Poly([1, 0, 0], field=GF2_1), [0], [2]),
    (galois.Poly([1, 0, 1, 1, 0], field=GF2_1), [0], [1]),
    (galois.Poly([1, 0, 0, 0], field=GF2_1), [0], [3]),
    (galois.Poly([1, 1], field=GF2_1), [1], [1]),
    (galois.Poly([1, 1, 1, 1, 0, 1, 0], field=GF2_1), [0], [1]),
    (galois.Poly([1, 0, 1, 0, 1, 0], field=GF2_1), [0], [1]),
    (galois.Poly([1, 0, 0, 0, 0, 1, 0, 1, 0, 0], field=GF2_1), [0], [2]),
    (galois.Poly([1, 0, 0, 0, 1, 0, 1, 1, 0, 0], field=GF2_1), [1, 0], [1, 2]),
    (galois.Poly([1, 0, 0, 1, 0, 1, 1], field=GF2_1), [1], [3]),
]

POLY_ROOTS_2_8 = [
    (galois.Poly([122, 72, 72, 190, 20, 85], field=GF2_8), [40], [1]),
    (galois.Poly([168, 190, 138, 248, 40, 130, 146], field=GF2_8), [170], [1]),
    (galois.Poly([43, 166, 112, 97, 87, 5, 70, 255], field=GF2_8), [30, 68, 195], [1, 1, 1]),
    (galois.Poly([156, 140, 16, 68, 87, 45, 29, 93, 84, 157, 23], field=GF2_8), [48], [1]),
    (galois.Poly([233, 60, 36, 133, 31, 6, 189], field=GF2_8), [75], [1]),
    (galois.Poly([28, 107, 119, 19, 199], field=GF2_8), [], []),
    (galois.Poly([30, 174, 229], field=GF2_8), [], []),
    (galois.Poly([85, 183, 152, 198, 4, 190, 237, 117, 23], field=GF2_8), [46, 152, 235], [1, 1, 1]),
    (galois.Poly([7, 31], field=GF2_8), [215], [1]),
    (galois.Poly([174, 124, 209, 174, 241, 126, 180], field=GF2_8), [], []),
    (galois.Poly([163], field=GF2_8), [], []),
    (galois.Poly([134, 7, 58, 0, 110, 125, 5, 53, 23], field=GF2_8), [], []),
    (galois.Poly([124, 139], field=GF2_8), [236], [1]),
    (galois.Poly([174, 102], field=GF2_8), [65], [1]),
    (galois.Poly([54, 119], field=GF2_8), [140], [1]),
    (galois.Poly([54, 144, 250, 29, 81], field=GF2_8), [], []),
    (galois.Poly([122, 5], field=GF2_8), [30], [1]),
    (galois.Poly([181, 179, 78, 5, 247, 60, 144, 3, 91, 78, 194], field=GF2_8), [50, 107, 245], [1, 1, 1]),
    (galois.Poly([224, 88, 133, 4, 171, 204], field=GF2_8), [], []),
    (galois.Poly([143], field=GF2_8), [], []),
]

POLY_ROOTS_3_1 = [
    (galois.Poly([1], field=GF3_1), [], []),
    (galois.Poly([1, 1, 2, 2, 2, 2, 1, 1], field=GF3_1), [1, 2], [2, 3]),
    (galois.Poly([1], field=GF3_1), [], []),
    (galois.Poly([2, 1], field=GF3_1), [1], [1]),
    (galois.Poly([1, 2, 0, 0, 0, 2], field=GF3_1), [2], [2]),
    (galois.Poly([2, 2, 1, 1, 2, 1, 0, 0, 2, 1, 1], field=GF3_1), [2], [1]),
    (galois.Poly([2, 0, 1, 2, 2, 1], field=GF3_1), [], []),
    (galois.Poly([1, 0, 0, 2], field=GF3_1), [1], [3]),
    (galois.Poly([1, 0, 0], field=GF3_1), [0], [2]),
    (galois.Poly([2, 1, 1, 2, 2, 1, 0, 0, 0], field=GF3_1), [1, 0], [1, 3]),
    (galois.Poly([1, 2, 1], field=GF3_1), [2], [2]),
    (galois.Poly([1, 0, 0, 0], field=GF3_1), [0], [3]),
    (galois.Poly([2, 1, 1, 0, 0, 1, 1, 2], field=GF3_1), [2], [5]),
    (galois.Poly([1, 2], field=GF3_1), [1], [1]),
    (galois.Poly([2, 0, 2], field=GF3_1), [], []),
    (galois.Poly([2, 1, 0, 0, 0, 2, 1, 1, 2, 2], field=GF3_1), [], []),
    (galois.Poly([1, 2, 1, 0, 0, 2, 2], field=GF3_1), [2], [1]),
    (galois.Poly([1, 0, 0, 0, 2, 1, 2, 0], field=GF3_1), [0, 1], [1, 1]),
    (galois.Poly([2, 1, 1, 2, 0], field=GF3_1), [0, 2, 1], [1, 1, 2]),
    (galois.Poly([1, 0, 2], field=GF3_1), [2, 1], [1, 1]),
]

POLY_ROOTS_3_5 = [
    (galois.Poly([137, 156, 158], field=GF3_5), [98, 155], [1, 1]),
    (galois.Poly([224, 110, 51, 222, 137, 101], field=GF3_5), [33], [1]),
    (galois.Poly([183, 2, 143], field=GF3_5), [205, 86], [1, 1]),
    (galois.Poly([228, 160, 55, 214, 125, 71, 185, 75, 108, 233], field=GF3_5), [], []),
    (galois.Poly([68, 116, 78, 119], field=GF3_5), [13, 240, 112], [1, 1, 1]),
    (galois.Poly([134, 92, 9, 129, 94, 218, 124, 62], field=GF3_5), [], []),
    (galois.Poly([211, 57, 219, 236], field=GF3_5), [], []),
    (galois.Poly([79, 215, 240, 201, 186], field=GF3_5), [128], [1]),
    (galois.Poly([185, 30, 77, 138, 27], field=GF3_5), [225], [1]),
    (galois.Poly([22], field=GF3_5), [], []),
    (galois.Poly([52, 167], field=GF3_5), [50], [1]),
    (galois.Poly([172, 28, 152, 76, 62, 23], field=GF3_5), [129], [1]),
    (galois.Poly([5, 21, 236, 16, 43, 37, 178, 98, 34, 35], field=GF3_5), [], []),
    (galois.Poly([64], field=GF3_5), [], []),
    (galois.Poly([12, 37, 131, 105, 74, 103, 140, 30], field=GF3_5), [65, 189, 147], [1, 1, 1]),
    (galois.Poly([62, 64], field=GF3_5), [242], [1]),
    (galois.Poly([68, 190, 187, 180, 71, 206, 166, 219, 79], field=GF3_5), [], []),
    (galois.Poly([235, 46, 207, 34, 7, 201, 120, 159, 96], field=GF3_5), [89], [1]),
    (galois.Poly([138, 46], field=GF3_5), [83], [1]),
    (galois.Poly([12, 189, 111, 238, 200], field=GF3_5), [16, 233, 196, 124], [1, 1, 1, 1]),
]

POLY_ROOTS_5_1 = [
    (galois.Poly([3, 0, 2, 3, 4, 2, 2, 1, 4, 4], field=GF5_1), [4, 1], [1, 1]),
    (galois.Poly([3, 0, 3, 0, 0], field=GF5_1), [3, 2, 0], [1, 1, 2]),
    (galois.Poly([1, 4, 4], field=GF5_1), [3], [2]),
    (galois.Poly([4, 4, 2, 0], field=GF5_1), [0, 3, 1], [1, 1, 1]),
    (galois.Poly([4, 4], field=GF5_1), [4], [1]),
    (galois.Poly([4, 0, 1, 0, 3, 3, 2], field=GF5_1), [3], [1]),
    (galois.Poly([2, 3, 4, 1, 1, 2, 3, 3], field=GF5_1), [], []),
    (galois.Poly([4], field=GF5_1), [], []),
    (galois.Poly([2, 2, 0, 4], field=GF5_1), [], []),
    (galois.Poly([4, 2, 1, 3, 1, 0], field=GF5_1), [0], [1]),
    (galois.Poly([3, 3, 4, 3], field=GF5_1), [], []),
    (galois.Poly([4, 4, 2, 4, 3, 1, 4, 2, 2], field=GF5_1), [2], [1]),
    (galois.Poly([4], field=GF5_1), [], []),
    (galois.Poly([4, 2, 3, 2, 3, 2], field=GF5_1), [2], [1]),
    (galois.Poly([4], field=GF5_1), [], []),
    (galois.Poly([2, 1, 3], field=GF5_1), [], []),
    (galois.Poly([3, 4, 4, 1, 1, 2, 1], field=GF5_1), [], []),
    (galois.Poly([2, 2, 3], field=GF5_1), [2], [2]),
    (galois.Poly([1, 4, 2, 4], field=GF5_1), [4], [1]),
    (galois.Poly([2, 3, 4], field=GF5_1), [], []),
]

POLY_ROOTS_5_4 = [
    (galois.Poly([362, 320], field=GF5_4), [147], [1]),
    (galois.Poly([215, 283, 164, 455, 92, 534, 170], field=GF5_4), [460, 304], [1, 1]),
    (galois.Poly([27, 503, 1, 608, 159, 412, 541, 35, 112], field=GF5_4), [], []),
    (galois.Poly([526, 22, 93], field=GF5_4), [], []),
    (galois.Poly([483, 258, 417, 1, 387, 141, 598, 32, 489, 111, 537], field=GF5_4), [39, 483], [1, 1]),
    (galois.Poly([44, 480, 61], field=GF5_4), [506, 623], [1, 1]),
    (galois.Poly([594, 221, 457, 247, 241, 367, 490, 40, 235, 418], field=GF5_4), [], []),
    (galois.Poly([37, 568, 308, 593, 193, 168, 467, 500, 198], field=GF5_4), [341], [1]),
    (galois.Poly([465, 389, 620, 592, 57, 358, 291, 505, 92], field=GF5_4), [599, 563], [1, 1]),
    (galois.Poly([85, 448, 537, 184, 181, 197, 518, 441, 372, 119], field=GF5_4), [20], [1]),
    (galois.Poly([517, 374, 125, 36, 544, 538, 561, 595, 406], field=GF5_4), [48, 247], [1, 1]),
    (galois.Poly([446, 330, 35, 369, 150, 191, 390], field=GF5_4), [13], [1]),
    (galois.Poly([203, 391, 540, 48, 536, 92, 184, 37, 279], field=GF5_4), [], []),
    (galois.Poly([45, 133, 380, 447, 104, 363], field=GF5_4), [], []),
    (galois.Poly([340, 216, 340, 256, 115, 448, 101, 234, 346, 134, 427], field=GF5_4), [417], [1]),
    (galois.Poly([28, 65, 84, 441, 143, 350, 90, 377, 154], field=GF5_4), [476, 270, 260, 175], [1, 1, 1, 1]),
    (galois.Poly([461, 560, 448, 83, 407, 220], field=GF5_4), [], []),
    (galois.Poly([421, 243, 444, 439], field=GF5_4), [], []),
    (galois.Poly([118, 27, 472, 23, 149, 11, 478, 85, 618, 5], field=GF5_4), [49, 309, 252], [1, 1, 2]),
    (galois.Poly([281, 598, 398, 156, 310, 427], field=GF5_4), [87, 562, 404], [1, 1, 1]),
]


def test_roots_exceptions():
    p = galois.Poly.Random(5)

    with pytest.raises(TypeError):
        p.roots(multiplicity=1)


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)])
def test_roots(characteristic, degree):
    LUT = eval(f"POLY_ROOTS_{characteristic}_{degree}")
    for item in LUT:
        poly, roots, multiplicities = item
        if len(roots) > 0:
            roots, multiplicities = zip(*sorted(zip(roots, multiplicities), key=lambda item: item[0]))  # Sort roots ascending
        r = poly.roots()
        assert np.array_equal(r, roots)


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)])
def test_roots_with_multiplicity(characteristic, degree):
    LUT = eval(f"POLY_ROOTS_{characteristic}_{degree}")
    for item in LUT:
        poly, roots, multiplicities = item
        if len(roots) > 0:
            roots, multiplicities = zip(*sorted(zip(roots, multiplicities), key=lambda item: item[0]))  # Sort roots ascending
        r, m = poly.roots(multiplicity=True)
        assert np.array_equal(r, roots)
        assert np.array_equal(m, multiplicities)
