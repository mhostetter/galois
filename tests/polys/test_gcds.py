"""
A pytest module to test computing GCDs and extended GCDs on polynomials over Galois fields.

Sage:
    def to_str(poly):
        c = poly.coefficients(sparse=False)[::-1]
        return f"galois.Poly({c}, field=GF{characteristic}_{degree})"

    N = 20
    for (characteristic, degree) in [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)]:
        print(f"POLY_EGCD_{characteristic}_{degree} = {{")
        R = GF(characteristic**degree, repr="int")["x"]
        for _ in range(N):
            a = R.random_element(randint(0, 10))
            b = R.random_element(randint(0, 10))
            d, s, t = xgcd(a, b)
            print(f"    ({to_str(a)}, {to_str(b)}): ({to_str(d)}, {to_str(s)}, {to_str(t)}),")
        print("}\n")
"""
import pytest

import galois

GF2_1 = galois.GF(2**1)
GF2_8 = galois.GF(2**8)
GF3_1 = galois.GF(3**1)
GF3_5 = galois.GF(3**5)
GF5_1 = galois.GF(5**1)
GF5_4 = galois.GF(5**4)

POLY_EGCD_2_1 = {
    (galois.Poly([1, 1, 1, 0, 0, 0], field=GF2_1), galois.Poly([1, 1, 0, 0], field=GF2_1)): (galois.Poly([1, 0, 0], field=GF2_1), galois.Poly([1], field=GF2_1), galois.Poly([1, 0, 1], field=GF2_1)),
    (galois.Poly([1, 1, 1, 0, 0], field=GF2_1), galois.Poly([1, 1, 1, 1, 1, 1, 0], field=GF2_1)): (galois.Poly([1, 1, 1, 0], field=GF2_1), galois.Poly([1, 0, 0], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 0, 1, 0, 0], field=GF2_1), galois.Poly([1, 1, 1, 0, 0, 0, 1, 1, 1, 0], field=GF2_1)): (galois.Poly([1, 0, 1, 0], field=GF2_1), galois.Poly([1, 1, 0, 1, 0, 1], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 0, 1, 0, 1, 0, 0, 0], field=GF2_1), galois.Poly([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], field=GF2_1)): (galois.Poly([1, 1, 1], field=GF2_1), galois.Poly([1, 0, 0, 1], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 1], field=GF2_1), galois.Poly([1, 1, 1, 0], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1, 0, 1], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1], field=GF2_1), galois.Poly([1], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 0, 0], field=GF2_1), galois.Poly([1, 1, 0, 0, 1, 0, 0, 0, 0], field=GF2_1)): (galois.Poly([1, 0, 0], field=GF2_1), galois.Poly([1], field=GF2_1), galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 0, 0, 0, 0, 0, 1, 0], field=GF2_1), galois.Poly([1, 1], field=GF2_1)): (galois.Poly([1, 1], field=GF2_1), galois.Poly([], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 0, 0, 0, 0, 1], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1, 0, 0, 0, 0], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 0, 1, 0, 1, 0, 0, 1, 0], field=GF2_1), galois.Poly([1, 0], field=GF2_1)): (galois.Poly([1, 0], field=GF2_1), galois.Poly([], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 0, 1, 0, 0, 1], field=GF2_1), galois.Poly([1, 0, 1, 0, 1, 0, 0, 1, 0], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1, 1, 1, 0, 1, 0, 1], field=GF2_1), galois.Poly([1, 1, 1, 0], field=GF2_1)),
    (galois.Poly([1, 0, 1, 0], field=GF2_1), galois.Poly([1], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([], field=GF2_1), galois.Poly([1], field=GF2_1)),
    (galois.Poly([1, 1, 0, 1, 0, 1], field=GF2_1), galois.Poly([1, 0], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1], field=GF2_1), galois.Poly([1, 1, 0, 1, 0], field=GF2_1)),
    (galois.Poly([1, 0, 1, 1], field=GF2_1), galois.Poly([1, 0, 1, 0, 0, 1, 1, 0, 1, 1], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1, 1, 0, 1, 1, 0, 0, 0, 1], field=GF2_1), galois.Poly([1, 1, 0], field=GF2_1)),
    (galois.Poly([1, 0, 0, 1, 1, 1, 1, 0, 1], field=GF2_1), galois.Poly([1, 0, 0, 0, 1, 1, 1, 1], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1, 1, 1, 1, 0, 0, 1], field=GF2_1), galois.Poly([1, 1, 1, 0, 1, 1, 0, 0], field=GF2_1)),
    (galois.Poly([1], field=GF2_1), galois.Poly([1, 0, 0], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1], field=GF2_1), galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 0, 1, 0, 1, 1, 1, 0], field=GF2_1), galois.Poly([1, 0, 1, 0], field=GF2_1)): (galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 0, 0, 0, 1, 1], field=GF2_1)),
    (galois.Poly([1, 0, 1, 1], field=GF2_1), galois.Poly([1, 1], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1], field=GF2_1), galois.Poly([1, 1, 0], field=GF2_1)),
    (galois.Poly([1], field=GF2_1), galois.Poly([1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0], field=GF2_1)): (galois.Poly([1], field=GF2_1), galois.Poly([1], field=GF2_1), galois.Poly([], field=GF2_1)),
    (galois.Poly([1, 0, 0, 1], field=GF2_1), galois.Poly([1, 1], field=GF2_1)): (galois.Poly([1, 1], field=GF2_1), galois.Poly([], field=GF2_1), galois.Poly([1], field=GF2_1)),
}

POLY_EGCD_2_8 = {
    (galois.Poly([162, 178, 132], field=GF2_8), galois.Poly([11, 8, 69, 189, 250], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([139, 164, 229, 128], field=GF2_8), galois.Poly([81, 95], field=GF2_8)),
    (galois.Poly([241, 135, 174, 133, 112], field=GF2_8), galois.Poly([94, 123, 179, 191, 246, 8, 80, 227, 129], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([245, 87, 175, 37, 230, 81, 190, 138], field=GF2_8), galois.Poly([253, 7, 169, 238], field=GF2_8)),
    (galois.Poly([253], field=GF2_8), galois.Poly([17, 200, 211, 118, 169, 0, 160, 21, 73, 234], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([255], field=GF2_8), galois.Poly([], field=GF2_8)),
    (galois.Poly([247], field=GF2_8), galois.Poly([87, 76, 75, 204, 176, 149, 69, 81, 23, 228, 72], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([201], field=GF2_8), galois.Poly([], field=GF2_8)),
    (galois.Poly([115, 142, 84, 97, 88, 190, 33], field=GF2_8), galois.Poly([10, 208, 186, 22], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([34, 87, 53], field=GF2_8), galois.Poly([162, 209, 106, 181, 204, 85], field=GF2_8)),
    (galois.Poly([183, 150, 104, 130, 89, 232, 143, 27, 220, 86], field=GF2_8), galois.Poly([133, 245, 200, 1, 218, 33, 70, 175, 176, 88, 159], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([177, 198, 140, 90, 186, 39, 131, 155, 210, 204], field=GF2_8), galois.Poly([248, 69, 55, 187, 105, 174, 108, 248, 99], field=GF2_8)),
    (galois.Poly([14, 135, 69, 101, 52], field=GF2_8), galois.Poly([226, 31, 27], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([20, 152], field=GF2_8), galois.Poly([228, 163, 232, 174], field=GF2_8)),
    (galois.Poly([47, 180, 224, 202, 194, 138, 149], field=GF2_8), galois.Poly([170, 212, 215, 117, 196, 164, 158, 43, 136, 92, 222], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([156, 140, 132, 238, 130, 191, 85, 231, 191, 223], field=GF2_8), galois.Poly([81, 186, 0, 156, 218, 96], field=GF2_8)),
    (galois.Poly([228, 125, 171, 186, 250, 122, 242, 189, 199, 164], field=GF2_8), galois.Poly([83, 216, 9, 184, 61, 90, 86, 27], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([253, 17, 26, 175, 92, 23, 220], field=GF2_8), galois.Poly([96, 231, 111, 199, 26, 191, 235, 251, 126], field=GF2_8)),
    (galois.Poly([77, 42, 225, 119, 226], field=GF2_8), galois.Poly([80, 90, 134, 116], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([82, 25, 22], field=GF2_8), galois.Poly([22, 38, 22, 198], field=GF2_8)),
    (galois.Poly([71, 136, 217, 56], field=GF2_8), galois.Poly([168, 232, 192, 48, 209, 243, 87, 137], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([82, 180, 235, 150, 157, 195, 70], field=GF2_8), galois.Poly([4, 232, 206], field=GF2_8)),
    (galois.Poly([181, 214, 155, 41, 218, 101, 196, 112], field=GF2_8), galois.Poly([22, 119, 167, 241, 188, 115, 96, 181, 88, 151, 9], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([85, 107, 206, 120, 209, 55, 6, 121, 172], field=GF2_8), galois.Poly([81, 63, 130, 181, 99, 147], field=GF2_8)),
    (galois.Poly([159, 40, 222, 215], field=GF2_8), galois.Poly([56, 37, 156, 146], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([214, 169, 152], field=GF2_8), galois.Poly([55, 221, 10], field=GF2_8)),
    (galois.Poly([8, 240, 147, 198], field=GF2_8), galois.Poly([20, 58, 181, 39, 142, 217, 18, 140, 214, 11], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([186, 191, 250, 207, 59, 12, 55, 85, 195], field=GF2_8), galois.Poly([29, 216, 178], field=GF2_8)),
    (galois.Poly([48, 152], field=GF2_8), galois.Poly([76, 42, 144], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([19, 128], field=GF2_8), galois.Poly([12], field=GF2_8)),
    (galois.Poly([132, 194, 170, 1, 165, 148, 248, 252, 31], field=GF2_8), galois.Poly([230, 165, 200, 77], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([85, 194, 74], field=GF2_8), galois.Poly([46, 46, 127, 177, 139, 191, 169, 71], field=GF2_8)),
    (galois.Poly([53, 31, 47, 112, 91, 176, 116], field=GF2_8), galois.Poly([27, 53, 25, 146, 92, 7], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([84, 140, 59, 39, 168], field=GF2_8), galois.Poly([87, 20, 21, 64, 245, 8], field=GF2_8)),
    (galois.Poly([15, 165], field=GF2_8), galois.Poly([118, 75], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([210], field=GF2_8), galois.Poly([135], field=GF2_8)),
    (galois.Poly([130, 39, 219, 53, 61, 176, 36, 73, 253], field=GF2_8), galois.Poly([251, 132, 217, 187, 210, 142, 132, 3, 143], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([63, 1, 134, 23, 237, 190, 82, 132], field=GF2_8), galois.Poly([151, 62, 187, 144, 166, 19, 32, 61], field=GF2_8)),
    (galois.Poly([138], field=GF2_8), galois.Poly([108, 255, 147, 139, 115], field=GF2_8)): (galois.Poly([1], field=GF2_8), galois.Poly([39], field=GF2_8), galois.Poly([], field=GF2_8)),
}

POLY_EGCD_3_1 = {
    (galois.Poly([1, 2, 1, 1, 2, 1, 1, 2, 2, 2], field=GF3_1), galois.Poly([1, 1, 1, 1, 1, 1, 2, 0, 1, 0], field=GF3_1)): (galois.Poly([1, 2], field=GF3_1), galois.Poly([1, 2, 0, 2, 0, 0, 1], field=GF3_1), galois.Poly([2, 0, 2, 0, 0, 1, 2], field=GF3_1)),
    (galois.Poly([2, 0, 2, 1, 0, 2, 0], field=GF3_1), galois.Poly([2, 1], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([1], field=GF3_1), galois.Poly([2, 2, 1, 2, 2, 1], field=GF3_1)),
    (galois.Poly([1, 2, 0, 0, 0], field=GF3_1), galois.Poly([2], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([], field=GF3_1), galois.Poly([2], field=GF3_1)),
    (galois.Poly([1, 1, 1, 0], field=GF3_1), galois.Poly([2, 2, 0, 2, 2, 2, 0, 0, 2], field=GF3_1)): (galois.Poly([1, 1, 1], field=GF3_1), galois.Poly([2, 0, 1, 1, 0, 1], field=GF3_1), galois.Poly([2], field=GF3_1)),
    (galois.Poly([2, 0, 0, 2, 0, 2, 1, 1], field=GF3_1), galois.Poly([1, 2, 2, 1, 1], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([1, 2, 0], field=GF3_1), galois.Poly([1, 0, 1, 1, 0, 1], field=GF3_1)),
    (galois.Poly([1, 1, 0], field=GF3_1), galois.Poly([2, 2, 1, 2, 1, 2, 2, 1], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([1, 0, 2, 2, 0, 1], field=GF3_1), galois.Poly([1], field=GF3_1)),
    (galois.Poly([2, 1, 2, 2, 2, 0, 0, 2, 1, 2], field=GF3_1), galois.Poly([1, 2, 1, 0, 0, 2], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([1, 2, 2, 1, 2], field=GF3_1), galois.Poly([1, 2, 2, 2, 0, 2, 0, 1, 0], field=GF3_1)),
    (galois.Poly([2, 0, 2, 2, 2, 0, 0, 0, 0, 0, 1], field=GF3_1), galois.Poly([2, 1, 0, 0, 1, 2, 0, 2, 0], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([2, 1, 2, 2, 1, 1, 1, 1], field=GF3_1), galois.Poly([1, 0, 2, 0, 1, 0, 2, 0, 1, 1], field=GF3_1)),
    (galois.Poly([1], field=GF3_1), galois.Poly([2, 2], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([1], field=GF3_1), galois.Poly([], field=GF3_1)),
    (galois.Poly([1, 2, 2, 1, 1, 1, 1], field=GF3_1), galois.Poly([1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([1, 2, 1, 1, 0, 1, 0, 2, 1], field=GF3_1), galois.Poly([2, 2, 2, 1, 0], field=GF3_1)),
    (galois.Poly([1, 0, 2, 2, 2], field=GF3_1), galois.Poly([2, 1, 2, 0, 0, 0, 1, 0], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([2, 1, 1, 0, 1, 2], field=GF3_1), galois.Poly([2, 0, 0], field=GF3_1)),
    (galois.Poly([2, 2, 0, 2, 1, 1, 2, 2, 1, 1], field=GF3_1), galois.Poly([1, 1, 0], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([2, 1], field=GF3_1), galois.Poly([2, 1, 0, 2, 0, 0, 1, 2, 0], field=GF3_1)),
    (galois.Poly([2, 1, 1], field=GF3_1), galois.Poly([1, 1, 1, 0, 1, 2, 2, 1, 1, 0], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([2, 0, 0, 2, 1, 0, 0, 0, 1], field=GF3_1), galois.Poly([2, 2], field=GF3_1)),
    (galois.Poly([1, 1, 1], field=GF3_1), galois.Poly([1, 2, 1, 2, 0, 0, 2], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([2, 1, 0, 2, 2, 2], field=GF3_1), galois.Poly([1, 1], field=GF3_1)),
    (galois.Poly([1, 1, 2, 0, 0, 1, 1], field=GF3_1), galois.Poly([1, 1, 1, 2, 2, 1, 1, 2, 2, 2], field=GF3_1)): (galois.Poly([1, 2], field=GF3_1), galois.Poly([2, 0, 0, 0, 1, 1, 2, 1], field=GF3_1), galois.Poly([1, 0, 1, 0, 2], field=GF3_1)),
    (galois.Poly([1, 1, 0, 0, 1], field=GF3_1), galois.Poly([1, 1, 1, 0], field=GF3_1)): (galois.Poly([1, 2], field=GF3_1), galois.Poly([2, 2], field=GF3_1), galois.Poly([1, 1, 2], field=GF3_1)),
    (galois.Poly([2, 0, 0, 1, 2, 0], field=GF3_1), galois.Poly([2, 2], field=GF3_1)): (galois.Poly([1, 1], field=GF3_1), galois.Poly([], field=GF3_1), galois.Poly([2], field=GF3_1)),
    (galois.Poly([1, 1, 0, 0, 2, 1, 2, 0, 0, 2], field=GF3_1), galois.Poly([2, 1, 2, 1, 1, 2, 2, 2], field=GF3_1)): (galois.Poly([1, 1, 2], field=GF3_1), galois.Poly([2, 2, 2, 0, 0], field=GF3_1), galois.Poly([2, 0, 2, 0, 1, 1, 1], field=GF3_1)),
    (galois.Poly([1, 2, 2, 0, 1, 2, 1, 2], field=GF3_1), galois.Poly([1, 1, 2, 1, 0, 0, 2], field=GF3_1)): (galois.Poly([1], field=GF3_1), galois.Poly([1, 2, 2, 2, 2, 2], field=GF3_1), galois.Poly([2, 0, 0, 0, 1, 0, 0], field=GF3_1)),
    (galois.Poly([1, 0, 2, 1, 1, 1, 0], field=GF3_1), galois.Poly([1, 1, 0, 1, 2, 2, 2, 0], field=GF3_1)): (galois.Poly([1, 2, 0], field=GF3_1), galois.Poly([2, 2, 2, 2, 2], field=GF3_1), galois.Poly([1, 0, 0, 0], field=GF3_1)),
}

POLY_EGCD_3_5 = {
    (galois.Poly([54, 147, 107, 183, 164, 182], field=GF3_5), galois.Poly([224, 183, 36, 173, 201, 210, 115, 68, 1, 155], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([177, 187, 54, 51, 60, 117, 239, 213, 214], field=GF3_5), galois.Poly([60, 191, 204, 155, 180], field=GF3_5)),
    (galois.Poly([140, 155, 117, 51, 204, 207], field=GF3_5), galois.Poly([145, 144, 62, 55, 3, 125], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([67, 110, 66, 193, 81], field=GF3_5), galois.Poly([216, 171, 193, 111, 165], field=GF3_5)),
    (galois.Poly([54, 149, 40, 207, 124, 140, 157, 104, 127], field=GF3_5), galois.Poly([138, 181, 97, 16, 183, 70, 179], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([168, 235, 183, 51, 210, 86], field=GF3_5), galois.Poly([146, 50, 155, 166, 140, 20, 5, 156], field=GF3_5)),
    (galois.Poly([209, 102], field=GF3_5), galois.Poly([110, 12], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([222], field=GF3_5), galois.Poly([20], field=GF3_5)),
    (galois.Poly([120, 133, 16, 111, 40, 189, 178, 194, 230, 128], field=GF3_5), galois.Poly([204, 182, 33, 155, 119, 11, 189], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([26, 187, 223, 80, 149, 30], field=GF3_5), galois.Poly([20, 68, 166, 40, 41, 23, 165, 39, 236], field=GF3_5)),
    (galois.Poly([237, 215, 188, 64, 53, 213, 22, 53], field=GF3_5), galois.Poly([114, 136, 82, 148, 184, 230, 5, 66, 195], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([216, 225, 116, 136, 5, 32, 199, 234], field=GF3_5), galois.Poly([145, 237, 167, 89, 41, 162, 8], field=GF3_5)),
    (galois.Poly([176], field=GF3_5), galois.Poly([175, 227, 29, 98, 236], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([63], field=GF3_5), galois.Poly([], field=GF3_5)),
    (galois.Poly([102], field=GF3_5), galois.Poly([230, 7, 227, 193, 39], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([155], field=GF3_5), galois.Poly([], field=GF3_5)),
    (galois.Poly([80, 230, 235, 202, 191, 56, 187, 169, 128], field=GF3_5), galois.Poly([122, 141, 14, 62, 209, 130, 91, 112, 65], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([109, 227, 202, 19, 169, 32, 84, 69], field=GF3_5), galois.Poly([229, 152, 94, 237, 27, 180, 68, 195], field=GF3_5)),
    (galois.Poly([241, 79, 131, 92, 91, 231, 83], field=GF3_5), galois.Poly([4, 229, 102], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([62, 233], field=GF3_5), galois.Poly([194, 44, 147, 31, 204, 190], field=GF3_5)),
    (galois.Poly([225, 44, 31, 24, 161, 207, 13], field=GF3_5), galois.Poly([99, 162], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([236], field=GF3_5), galois.Poly([232, 205, 2, 54, 102, 155], field=GF3_5)),
    (galois.Poly([223, 33, 67, 159, 128, 173], field=GF3_5), galois.Poly([53, 82, 67, 6, 93, 179], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([40, 235, 111, 45, 205], field=GF3_5), galois.Poly([42, 185, 12, 209, 81], field=GF3_5)),
    (galois.Poly([18, 226, 234], field=GF3_5), galois.Poly([190, 66], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([192], field=GF3_5), galois.Poly([31, 65], field=GF3_5)),
    (galois.Poly([239, 161, 221, 205, 121, 112, 142, 99, 4], field=GF3_5), galois.Poly([100, 97, 48, 182, 43], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([205, 125, 10, 171], field=GF3_5), galois.Poly([93, 70, 89, 19, 183, 135, 229, 96], field=GF3_5)),
    (galois.Poly([70, 223, 129, 82, 170, 164, 116, 35, 133], field=GF3_5), galois.Poly([127, 25, 89, 241, 142, 81, 48, 169, 2, 46, 127], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([209, 24, 209, 58, 44, 229, 12, 86, 156, 101], field=GF3_5), galois.Poly([116, 183, 57, 50, 207, 174, 146, 65], field=GF3_5)),
    (galois.Poly([211, 133, 46, 57, 117, 236, 67, 60, 130, 4, 159], field=GF3_5), galois.Poly([186, 133, 43, 195, 61, 54, 112, 34, 99, 57], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([117, 27, 46, 179, 115, 227, 37, 108, 190], field=GF3_5), galois.Poly([123, 237, 7, 180, 58, 131, 183, 21, 157, 156], field=GF3_5)),
    (galois.Poly([154, 239, 3, 13, 197, 0, 211, 242, 232], field=GF3_5), galois.Poly([173, 108], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([99], field=GF3_5), galois.Poly([16, 19, 222, 239, 50, 129, 170, 66], field=GF3_5)),
    (galois.Poly([117, 6, 103, 103], field=GF3_5), galois.Poly([64, 58, 214, 126, 152, 132, 69, 112], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([175, 9, 182, 144, 117, 102, 64], field=GF3_5), galois.Poly([133, 117, 45], field=GF3_5)),
    (galois.Poly([188, 120, 67, 193, 105, 91, 89, 138, 129], field=GF3_5), galois.Poly([130, 34, 232, 103, 98, 48, 87, 240, 15], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([4, 123, 210, 204, 216, 169, 164, 100], field=GF3_5), galois.Poly([224, 233, 85, 178, 142, 30, 34, 201], field=GF3_5)),
    (galois.Poly([12, 126, 14, 125, 209, 27, 24, 88, 104, 61, 118], field=GF3_5), galois.Poly([199], field=GF3_5)): (galois.Poly([1], field=GF3_5), galois.Poly([], field=GF3_5), galois.Poly([177], field=GF3_5)),
}

POLY_EGCD_5_1 = {
    (galois.Poly([3, 3, 2, 1], field=GF5_1), galois.Poly([2, 3, 1, 3], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([4, 1, 1], field=GF5_1), galois.Poly([4, 4, 0], field=GF5_1)),
    (galois.Poly([3, 3], field=GF5_1), galois.Poly([1, 2, 3], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([4, 4], field=GF5_1), galois.Poly([3], field=GF5_1)),
    (galois.Poly([4, 1, 2, 4], field=GF5_1), galois.Poly([3, 1, 2, 2, 0, 4, 2], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([2, 3, 2, 3, 4, 1], field=GF5_1), galois.Poly([4, 4, 1], field=GF5_1)),
    (galois.Poly([4, 3], field=GF5_1), galois.Poly([4, 1, 3, 0, 3, 3, 2, 4], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([1, 2, 3, 4, 4, 4, 0], field=GF5_1), galois.Poly([4], field=GF5_1)),
    (galois.Poly([3, 3, 2, 4, 4, 2, 3, 2], field=GF5_1), galois.Poly([4, 0, 3, 4, 0, 1, 1, 2, 0, 1], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([3, 2, 1, 2, 2, 1, 0, 0], field=GF5_1), galois.Poly([4, 0, 2, 1, 0, 1], field=GF5_1)),
    (galois.Poly([3, 4, 4, 2, 3, 1, 0], field=GF5_1), galois.Poly([4, 1, 1, 4, 4, 3, 0, 0, 3, 3, 1], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([4, 2, 2, 4, 2, 1, 3, 2, 2, 2], field=GF5_1), galois.Poly([2, 4, 1, 4, 0, 1], field=GF5_1)),
    (galois.Poly([3, 3, 2, 4, 0, 0], field=GF5_1), galois.Poly([2, 2, 2, 3, 0, 1, 1, 0, 2], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([2, 3, 1, 4, 0, 3, 3, 2], field=GF5_1), galois.Poly([2, 3, 2, 0, 3], field=GF5_1)),
    (galois.Poly([3], field=GF5_1), galois.Poly([3, 4, 2, 3, 3, 0, 2, 2], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([2], field=GF5_1), galois.Poly([], field=GF5_1)),
    (galois.Poly([2, 1, 3, 4, 4, 1, 3, 2], field=GF5_1), galois.Poly([4, 2, 4, 2, 1, 0], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([4, 0, 3, 3], field=GF5_1), galois.Poly([3, 0, 0, 1, 3, 0], field=GF5_1)),
    (galois.Poly([4, 3, 1, 2, 3, 2, 3], field=GF5_1), galois.Poly([3, 1, 0, 3, 3, 1, 3, 0, 4], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([1, 3, 1, 1, 1, 4, 0, 0], field=GF5_1), galois.Poly([2, 1, 0, 4, 0, 4], field=GF5_1)),
    (galois.Poly([3], field=GF5_1), galois.Poly([2, 2], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([2], field=GF5_1), galois.Poly([], field=GF5_1)),
    (galois.Poly([1, 2], field=GF5_1), galois.Poly([1, 2, 1, 1, 2], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([1, 0, 1, 4], field=GF5_1), galois.Poly([4], field=GF5_1)),
    (galois.Poly([3, 1], field=GF5_1), galois.Poly([1, 2, 2, 0, 4], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([4, 0, 3, 4], field=GF5_1), galois.Poly([3], field=GF5_1)),
    (galois.Poly([2, 0, 2, 2, 0], field=GF5_1), galois.Poly([4, 3], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([1], field=GF5_1), galois.Poly([2, 1, 0, 2], field=GF5_1)),
    (galois.Poly([4, 3, 3, 2], field=GF5_1), galois.Poly([3, 0, 4, 4], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([2, 2, 2], field=GF5_1), galois.Poly([4, 2, 3], field=GF5_1)),
    (galois.Poly([1, 1, 0], field=GF5_1), galois.Poly([3, 3, 0, 1, 1, 1, 4, 0, 0, 2], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([3, 1, 0, 1, 2, 1, 0, 3, 2], field=GF5_1), galois.Poly([4, 3], field=GF5_1)),
    (galois.Poly([2, 0], field=GF5_1), galois.Poly([1, 2, 4, 1, 4, 3, 4, 0, 2, 0], field=GF5_1)): (galois.Poly([1, 0], field=GF5_1), galois.Poly([3], field=GF5_1), galois.Poly([], field=GF5_1)),
    (galois.Poly([1, 3, 2, 1, 0], field=GF5_1), galois.Poly([3, 3, 0], field=GF5_1)): (galois.Poly([1, 0], field=GF5_1), galois.Poly([1], field=GF5_1), galois.Poly([3, 1, 0], field=GF5_1)),
    (galois.Poly([3, 3, 3, 2, 3, 0, 1, 0, 4, 3], field=GF5_1), galois.Poly([2, 0, 4, 3, 0, 3, 4, 4, 1, 3, 2], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([1, 0, 0, 1, 0, 0, 1, 2, 0], field=GF5_1), galois.Poly([1, 1, 4, 4, 0, 3, 0, 3], field=GF5_1)),
    (galois.Poly([4, 1, 0, 2, 2, 0, 0, 0], field=GF5_1), galois.Poly([4, 2, 4, 2, 2, 3, 1], field=GF5_1)): (galois.Poly([1], field=GF5_1), galois.Poly([3, 3, 4, 1, 1, 2], field=GF5_1), galois.Poly([2, 4, 0, 4, 2, 2, 1], field=GF5_1)),
}

POLY_EGCD_5_4 = {
    (galois.Poly([403, 70, 65, 446, 191, 484, 232, 323, 602, 254], field=GF5_4), galois.Poly([444, 350, 76, 76, 500, 621, 211, 194, 475, 189], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([192, 143, 498, 425, 282, 223, 162, 521, 381], field=GF5_4), galois.Poly([169, 549, 460, 18, 155, 481, 548, 389, 393], field=GF5_4)),
    (galois.Poly([75, 571, 551, 190, 439, 235, 445], field=GF5_4), galois.Poly([290, 586], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([369], field=GF5_4), galois.Poly([32, 412, 221, 190, 176, 310], field=GF5_4)),
    (galois.Poly([594, 385, 548, 399, 269, 445, 459, 185, 67, 170, 114], field=GF5_4), galois.Poly([4, 70, 366, 294, 593, 410, 156, 110], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([586, 542, 371, 214, 373, 428, 623], field=GF5_4), galois.Poly([94, 183, 386, 563, 429, 20, 354, 116, 232, 41], field=GF5_4)),
    (galois.Poly([144, 299, 484, 465, 334, 452, 1, 170, 592], field=GF5_4), galois.Poly([592, 105, 194, 262], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([437, 49, 564], field=GF5_4), galois.Poly([448, 89, 257, 574, 304, 323, 98, 552], field=GF5_4)),
    (galois.Poly([447, 281, 273, 248, 594, 453, 292], field=GF5_4), galois.Poly([611, 8, 193, 350, 269, 515, 370, 38, 560], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([255, 68, 95, 207, 426, 245, 531, 55], field=GF5_4), galois.Poly([318, 510, 9, 235, 306, 494], field=GF5_4)),
    (galois.Poly([145, 429], field=GF5_4), galois.Poly([155], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([], field=GF5_4), galois.Poly([465], field=GF5_4)),
    (galois.Poly([424, 481, 136, 33, 550, 57], field=GF5_4), galois.Poly([457, 16, 450, 463, 502, 193, 225], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([146, 352, 26, 125, 186, 310], field=GF5_4), galois.Poly([608, 226, 325, 38, 578], field=GF5_4)),
    (galois.Poly([357, 192, 498, 219, 543, 351, 459, 102, 331, 502], field=GF5_4), galois.Poly([276, 573, 446, 200, 394, 111, 427, 14, 50], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([280, 319, 185, 309, 577, 495, 623, 215], field=GF5_4), galois.Poly([483, 614, 226, 305, 234, 335, 210, 544, 203], field=GF5_4)),
    (galois.Poly([295, 239, 450, 506, 54, 588, 74, 409], field=GF5_4), galois.Poly([265, 451], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([622], field=GF5_4), galois.Poly([410, 136, 472, 568, 48, 329, 507], field=GF5_4)),
    (galois.Poly([604], field=GF5_4), galois.Poly([279, 146, 619, 392, 358, 433, 299, 7, 139, 418, 314], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([429], field=GF5_4), galois.Poly([], field=GF5_4)),
    (galois.Poly([46, 175, 271, 435, 400, 310, 78, 282, 373, 395], field=GF5_4), galois.Poly([64, 512, 236, 149, 281, 325], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([323, 222, 244, 509, 565], field=GF5_4), galois.Poly([433, 398, 596, 365, 510, 102, 419, 537, 151], field=GF5_4)),
    (galois.Poly([153, 585, 483, 4, 393, 527, 224, 365, 623, 39], field=GF5_4), galois.Poly([573, 101], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([383], field=GF5_4), galois.Poly([96, 39, 328, 565, 303, 137, 504, 201, 34], field=GF5_4)),
    (galois.Poly([162, 203, 230, 295, 481], field=GF5_4), galois.Poly([338, 364, 327, 342, 285, 98, 92], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([325, 160, 90, 123, 177, 324], field=GF5_4), galois.Poly([522, 25, 144, 467], field=GF5_4)),
    (galois.Poly([123, 583, 75, 252, 507, 621, 161, 203, 335, 545], field=GF5_4), galois.Poly([164], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([], field=GF5_4), galois.Poly([274], field=GF5_4)),
    (galois.Poly([497, 188, 82], field=GF5_4), galois.Poly([34, 305, 158, 286, 353], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([546, 596, 158, 562], field=GF5_4), galois.Poly([411, 284], field=GF5_4)),
    (galois.Poly([185, 120, 452, 107, 253], field=GF5_4), galois.Poly([570, 124, 286, 535, 597, 487, 198, 22], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([269, 70, 220, 555, 435, 446, 369], field=GF5_4), galois.Poly([180, 281, 3, 211], field=GF5_4)),
    (galois.Poly([216, 1], field=GF5_4), galois.Poly([596, 3, 131, 14, 603, 77, 49, 389, 275, 108], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([522, 413, 7, 388, 399, 192, 211, 111, 387], field=GF5_4), galois.Poly([158], field=GF5_4)),
    (galois.Poly([255, 331, 507, 183, 124, 368, 440, 308, 590, 10, 572], field=GF5_4), galois.Poly([190, 261, 462, 31, 297], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([279, 348, 234, 260], field=GF5_4), galois.Poly([92, 604, 110, 493, 224, 497, 527, 128, 108, 536], field=GF5_4)),
    (galois.Poly([116, 367], field=GF5_4), galois.Poly([618, 119, 29, 8, 276, 93, 246, 601, 202], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([561, 621, 503, 623, 65, 351, 245, 276], field=GF5_4), galois.Poly([327], field=GF5_4)),
    (galois.Poly([93], field=GF5_4), galois.Poly([559, 77, 20, 70, 379, 372, 143, 267], field=GF5_4)): (galois.Poly([1], field=GF5_4), galois.Poly([183], field=GF5_4), galois.Poly([], field=GF5_4)),
}


def test_poly_gcd_exceptions():
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])

    with pytest.raises(TypeError):
        galois.poly_gcd(a.coeffs, b)
    with pytest.raises(TypeError):
        galois.poly_gcd(a, b.coeffs)
    with pytest.raises(ValueError):
        galois.poly_gcd(a, galois.Poly(b.coeffs, field=galois.GF(3)))


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)])
def test_poly_gcd(characteristic, degree):
    LUT = eval(f"POLY_EGCD_{characteristic}_{degree}")
    for key in LUT:
        a, b = key
        gcd = LUT[key][0]
        assert galois.poly_gcd(a, b) == gcd


def test_poly_egcd_exceptions():
    a = galois.Poly.Degrees([10, 9, 8, 6, 5, 4, 0])
    b = galois.Poly.Degrees([9, 6, 5, 3, 2, 0])

    with pytest.raises(TypeError):
        galois.poly_egcd(a.coeffs, b)
    with pytest.raises(TypeError):
        galois.poly_egcd(a, b.coeffs)
    with pytest.raises(ValueError):
        galois.poly_egcd(a, galois.Poly(b.coeffs, field=galois.GF(3)))


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)])
def test_poly_egcd(characteristic, degree):
    LUT = eval(f"POLY_EGCD_{characteristic}_{degree}")
    for key in LUT:
        a, b = key
        gcd, s, t = LUT[key]
        assert galois.poly_egcd(a, b) == (gcd, s, t)
