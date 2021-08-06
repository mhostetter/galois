"""
A pytest module to test factoring polynomials over Galois fields.

Sage:
    def to_str(poly):
        c = poly.coefficients(sparse=False)[::-1]
        return f"galois.Poly({c}, field=GF{characteristic}_{degree})"

    N = 20
    for (characteristic, degree) in [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)]:
        print(f"POLY_FACTORS_{characteristic}_{degree} = {{")
        R = GF(characteristic**degree, repr="int")["x"]
        for _ in range(N):
            a = R.random_element(randint(0, 20))
            polys = []
            exponents = []
            for item in factor(a):
                polys.append(to_str(item[0]))
                exponents.append(item[1])
            print(f"    {to_str(a)}: ([{', '.join(polys)}], {exponents}),")
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

POLY_FACTORS_2_1 = {
    galois.Poly([1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], field=GF2_1): ([galois.Poly([1, 1], field=GF2_1), galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 0, 1, 1, 1], field=GF2_1)], [4, 8, 1]),
    galois.Poly([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], field=GF2_1): ([galois.Poly([1, 0, 0, 1, 1, 1, 0, 1], field=GF2_1), galois.Poly([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1], field=GF2_1)], [1, 1]),
    galois.Poly([1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 0, 0, 1], field=GF2_1), galois.Poly([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1], field=GF2_1)], [2, 1, 1]),
    galois.Poly([1, 1, 1, 1, 1, 0, 0, 0, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 1, 1, 1], field=GF2_1)], [4, 1]),
    galois.Poly([1, 1, 1, 0, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 1], field=GF2_1)], [2, 1]),
    galois.Poly([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1], field=GF2_1): ([galois.Poly([1, 0, 0, 1, 0, 1], field=GF2_1), galois.Poly([1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1], field=GF2_1)], [1, 1]),
    galois.Poly([1, 0, 1, 1, 1, 1, 0, 1, 1], field=GF2_1): ([galois.Poly([1, 0, 1, 1, 1, 1, 0, 1, 1], field=GF2_1)], [1]),
    galois.Poly([1, 0, 1, 1, 0, 1, 0, 0, 1, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 0, 1, 1, 0, 1, 0, 0, 1], field=GF2_1)], [1, 1]),
    galois.Poly([1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1], field=GF2_1): ([galois.Poly([1, 1, 1], field=GF2_1), galois.Poly([1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1], field=GF2_1)], [1, 1]),
    galois.Poly([1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1], field=GF2_1): ([galois.Poly([1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1], field=GF2_1)], [1]),
    galois.Poly([1, 0, 0, 0, 1, 1], field=GF2_1): ([galois.Poly([1, 1, 1], field=GF2_1), galois.Poly([1, 1, 0, 1], field=GF2_1)], [1, 1]),
    galois.Poly([1, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1)], [1]),
    galois.Poly([1, 1, 1, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 1], field=GF2_1)], [1, 1]),
    galois.Poly([1, 1, 0, 0, 0, 1, 1, 0, 0, 0], field=GF2_1): ([galois.Poly([1, 1], field=GF2_1), galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 1, 1, 1], field=GF2_1)], [2, 3, 1]),
    galois.Poly([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1], field=GF2_1)], [1, 1]),
    galois.Poly([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], field=GF2_1): ([galois.Poly([1, 1], field=GF2_1), galois.Poly([1, 1, 1], field=GF2_1), galois.Poly([1, 0, 1, 1, 0, 1, 1], field=GF2_1)], [7, 1, 1]),
    galois.Poly([1, 0, 0], field=GF2_1): ([galois.Poly([1, 0], field=GF2_1)], [2]),
    galois.Poly([1, 1, 1, 0, 0, 0, 1, 0, 0], field=GF2_1): ([galois.Poly([1, 1], field=GF2_1), galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 0, 1, 1, 1, 1], field=GF2_1)], [1, 2, 1]),
    galois.Poly([1, 0, 0, 0, 1], field=GF2_1): ([galois.Poly([1, 1], field=GF2_1)], [4]),
    galois.Poly([1, 0, 0, 1, 0, 0], field=GF2_1): ([galois.Poly([1, 1], field=GF2_1), galois.Poly([1, 0], field=GF2_1), galois.Poly([1, 1, 1], field=GF2_1)], [1, 2, 1]),
}

POLY_FACTORS_2_8 = {
    galois.Poly([22, 218, 114, 5, 228, 244, 182, 63, 40, 154, 159], field=GF2_8): ([galois.Poly([1, 150], field=GF2_8), galois.Poly([1, 49, 249], field=GF2_8), galois.Poly([1, 152, 93], field=GF2_8), galois.Poly([1, 106, 250, 108, 93, 236], field=GF2_8)], [1, 1, 1, 1]),
    galois.Poly([77, 149, 138], field=GF2_8): ([galois.Poly([1, 83], field=GF2_8), galois.Poly([1, 102], field=GF2_8)], [1, 1]),
    galois.Poly([23, 136, 197, 198, 78, 75, 220], field=GF2_8): ([galois.Poly([1, 205], field=GF2_8), galois.Poly([1, 104, 80], field=GF2_8), galois.Poly([1, 223, 60, 131], field=GF2_8)], [1, 1, 1]),
    galois.Poly([155, 53, 216, 173, 15, 221, 85, 225, 1, 83, 37, 107, 1, 187, 163, 62, 194, 229, 122], field=GF2_8): ([galois.Poly([1, 98], field=GF2_8), galois.Poly([1, 158], field=GF2_8), galois.Poly([1, 192, 31, 120, 11, 48, 11, 208, 194, 106, 132, 201, 8, 186, 241, 64, 224], field=GF2_8)], [1, 1, 1]),
    galois.Poly([31, 3, 92], field=GF2_8): ([galois.Poly([1, 126, 45], field=GF2_8)], [1]),
    galois.Poly([218, 165, 147, 140, 70, 60, 111, 98], field=GF2_8): ([galois.Poly([1, 62], field=GF2_8), galois.Poly([1, 154], field=GF2_8), galois.Poly([1, 209], field=GF2_8), galois.Poly([1, 37, 22, 191, 158], field=GF2_8)], [1, 1, 1, 1]),
    galois.Poly([160, 176, 167, 176], field=GF2_8): ([galois.Poly([1, 54], field=GF2_8), galois.Poly([1, 234, 25], field=GF2_8)], [1, 1]),
    galois.Poly([228, 41, 89, 139, 144, 49, 188], field=GF2_8): ([galois.Poly([1, 34, 202], field=GF2_8), galois.Poly([1, 237, 177, 149, 175], field=GF2_8)], [1, 1]),
    galois.Poly([128, 150, 224, 45, 195, 44, 118, 77], field=GF2_8): ([galois.Poly([1, 95, 141], field=GF2_8), galois.Poly([1, 169, 75, 169, 218, 200], field=GF2_8)], [1, 1]),
    galois.Poly([155, 166, 22, 196, 215, 200, 73, 4], field=GF2_8): ([galois.Poly([1, 233, 234, 69, 81, 251, 174, 106], field=GF2_8)], [1]),
    galois.Poly([8, 198, 224, 186, 222, 243, 53, 17, 202, 252, 190, 112, 155, 61, 172, 148, 117, 183, 247, 31, 58], field=GF2_8): ([galois.Poly([1, 246, 24, 158], field=GF2_8), galois.Poly([1, 99, 51, 65, 222], field=GF2_8), galois.Poly([1, 118, 240, 214, 199, 248], field=GF2_8), galois.Poly([1, 50, 89, 131, 92, 136, 109, 177, 56], field=GF2_8)], [1, 1, 1, 1]),
    galois.Poly([53, 3, 38], field=GF2_8): ([galois.Poly([1, 56], field=GF2_8), galois.Poly([1, 96], field=GF2_8)], [1, 1]),
    galois.Poly([83, 240, 23, 238, 185, 148, 158, 104, 250, 2, 19, 63, 247, 113, 14, 112, 44, 208], field=GF2_8): ([galois.Poly([1, 138], field=GF2_8), galois.Poly([1, 177], field=GF2_8), galois.Poly([1, 228], field=GF2_8), galois.Poly([1, 141, 75], field=GF2_8), galois.Poly([1, 66, 71, 4, 42, 45, 182], field=GF2_8), galois.Poly([1, 149, 95, 175, 72, 149, 110], field=GF2_8)], [1, 1, 1, 1, 1, 1]),
    galois.Poly([96, 249, 52, 181, 42, 25, 102, 232, 216], field=GF2_8): ([galois.Poly([1, 120], field=GF2_8), galois.Poly([1, 237, 122, 217, 112, 96, 9, 84], field=GF2_8)], [1, 1]),
    galois.Poly([185, 5, 87, 247, 199, 65, 139, 77, 32, 115, 162, 153, 139], field=GF2_8): ([galois.Poly([1, 247], field=GF2_8), galois.Poly([1, 30, 63, 47, 246, 9, 199, 52, 167, 228, 201, 28], field=GF2_8)], [1, 1]),
    galois.Poly([27, 12, 40, 84, 4, 102, 236, 144, 142, 245, 66], field=GF2_8): ([galois.Poly([1, 44], field=GF2_8), galois.Poly([1, 63], field=GF2_8), galois.Poly([1, 93, 77, 128, 155, 47, 179, 96, 108], field=GF2_8)], [1, 1, 1]),
    galois.Poly([171, 108, 125, 250, 233, 42, 55, 161, 118, 114, 97, 190, 56, 63, 197, 70, 239, 29, 249, 77, 122], field=GF2_8): ([galois.Poly([1, 163], field=GF2_8), galois.Poly([1, 198, 191, 38, 246, 108, 49, 28, 34, 117, 213, 49, 81, 147, 69, 137, 219, 81, 203, 239], field=GF2_8)], [1, 1]),
    galois.Poly([36, 246, 200, 41, 119, 40, 32, 103, 224, 108, 116], field=GF2_8): ([galois.Poly([1, 6, 52, 144, 179], field=GF2_8), galois.Poly([1, 230, 18, 207, 86, 43, 151], field=GF2_8)], [1, 1]),
    galois.Poly([3, 29, 34, 99, 102, 50, 178, 76, 219, 195, 240, 118], field=GF2_8): ([galois.Poly([1, 175, 4], field=GF2_8), galois.Poly([1, 190, 10], field=GF2_8), galois.Poly([1, 113, 0, 16], field=GF2_8), galois.Poly([1, 107, 163, 95, 53], field=GF2_8)], [1, 1, 1, 1]),
    galois.Poly([251, 134, 21, 235, 99, 148, 17, 154, 115, 135, 22, 100, 183], field=GF2_8): ([galois.Poly([1, 188], field=GF2_8), galois.Poly([1, 102, 254], field=GF2_8), galois.Poly([1, 112, 180], field=GF2_8), galois.Poly([1, 145, 240, 95, 82, 58, 41, 1], field=GF2_8)], [1, 1, 1, 1]),
}

POLY_FACTORS_3_1 = {
    galois.Poly([1, 0, 2, 2, 1, 2, 2, 0, 1, 2], field=GF3_1): ([galois.Poly([1, 2, 1, 2, 1], field=GF3_1), galois.Poly([1, 1, 2, 1, 0, 2], field=GF3_1)], [1, 1]),
    galois.Poly([1, 0, 0, 2, 1, 1, 2, 1, 0, 1, 0, 1, 1, 2, 2, 2, 0], field=GF3_1): ([galois.Poly([1, 0], field=GF3_1), galois.Poly([1, 1], field=GF3_1), galois.Poly([1, 2, 2, 2, 1, 2], field=GF3_1), galois.Poly([1, 0, 2, 1, 2, 1, 0, 1, 1, 1], field=GF3_1)], [1, 1, 1, 1]),
    galois.Poly([2, 2], field=GF3_1): ([galois.Poly([1, 1], field=GF3_1)], [1]),
    galois.Poly([1, 1, 0, 2, 1, 1, 1], field=GF3_1): ([galois.Poly([1, 1, 0, 2, 1, 1, 1], field=GF3_1)], [1]),
    galois.Poly([1, 1, 1, 0], field=GF3_1): ([galois.Poly([1, 0], field=GF3_1), galois.Poly([1, 2], field=GF3_1)], [1, 2]),
    galois.Poly([1, 2, 1, 1], field=GF3_1): ([galois.Poly([1, 2, 1, 1], field=GF3_1)], [1]),
    galois.Poly([1, 1, 2, 1, 2, 1, 0, 1, 0, 2, 1, 0, 2, 2, 1, 1, 1, 2, 0, 1], field=GF3_1): ([galois.Poly([1, 2, 2], field=GF3_1), galois.Poly([1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 2, 1, 2], field=GF3_1)], [1, 1]),
    galois.Poly([1, 1, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 2, 0, 1, 0, 0, 2], field=GF3_1): ([galois.Poly([1, 2, 1, 0, 1], field=GF3_1), galois.Poly([1, 2, 0, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 2], field=GF3_1)], [1, 1]),
    galois.Poly([1, 1, 2, 1, 2, 1, 2, 2, 0, 2, 0, 0, 2, 1, 1, 1, 2, 2, 1], field=GF3_1): ([galois.Poly([1, 2], field=GF3_1), galois.Poly([1, 2, 2, 1, 0, 2], field=GF3_1), galois.Poly([1, 0, 2, 0, 0, 1, 2, 2, 0, 2, 0, 0, 1], field=GF3_1)], [1, 1, 1]),
    galois.Poly([2, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0, 2], field=GF3_1): ([galois.Poly([1, 1, 2, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 0, 1], field=GF3_1)], [1]),
    galois.Poly([2, 2, 0, 1, 0, 1, 2, 2, 2, 2, 0, 2, 0, 2, 0, 1, 0, 2, 1, 0, 0], field=GF3_1): ([galois.Poly([1, 0], field=GF3_1), galois.Poly([1, 1, 2, 0, 0, 2, 0, 1], field=GF3_1), galois.Poly([1, 0, 1, 1, 0, 1, 0, 2, 0, 2, 1, 2], field=GF3_1)], [2, 1, 1]),
    galois.Poly([2, 1, 1, 0, 2, 1, 1], field=GF3_1): ([galois.Poly([1, 1, 2], field=GF3_1), galois.Poly([1, 2, 2], field=GF3_1)], [1, 2]),
    galois.Poly([2, 1, 2, 0, 0, 1, 2, 1, 2, 2, 2], field=GF3_1): ([galois.Poly([1, 2], field=GF3_1), galois.Poly([1, 1, 1, 0, 1], field=GF3_1), galois.Poly([1, 2, 1, 1, 1, 2], field=GF3_1)], [1, 1, 1]),
    galois.Poly([2, 0, 1, 0, 2, 0, 2, 2, 2, 0, 0, 1, 1, 0], field=GF3_1): ([galois.Poly([1, 0], field=GF3_1), galois.Poly([1, 0, 2, 0, 1, 0, 1, 1, 1, 0, 0, 2, 2], field=GF3_1)], [1, 1]),
    galois.Poly([1, 2], field=GF3_1): ([galois.Poly([1, 2], field=GF3_1)], [1]),
    galois.Poly([1, 1, 0, 2, 2, 1, 1, 1, 2, 2], field=GF3_1): ([galois.Poly([1, 1, 0, 2, 2, 1, 1, 1, 2, 2], field=GF3_1)], [1]),
    galois.Poly([2, 0, 2, 1, 1, 2, 0, 2, 2, 0, 1, 1, 1, 2, 0, 2], field=GF3_1): ([galois.Poly([1, 0, 0, 1, 1, 2], field=GF3_1), galois.Poly([1, 1, 0, 1, 1, 1], field=GF3_1), galois.Poly([1, 2, 2, 1, 0, 2], field=GF3_1)], [1, 1, 1]),
    galois.Poly([2, 0, 2, 1, 0, 1, 0, 1], field=GF3_1): ([galois.Poly([1, 0, 1, 2, 0, 2, 0, 2], field=GF3_1)], [1]),
    galois.Poly([2, 1, 1, 0], field=GF3_1): ([galois.Poly([1, 0], field=GF3_1), galois.Poly([1, 2, 2], field=GF3_1)], [1, 1]),
    galois.Poly([1, 0, 1], field=GF3_1): ([galois.Poly([1, 0, 1], field=GF3_1)], [1]),
}

POLY_FACTORS_3_5 = {
    galois.Poly([201, 48, 172, 3, 178, 144, 124], field=GF3_5): ([galois.Poly([1, 30], field=GF3_5), galois.Poly([1, 161], field=GF3_5), galois.Poly([1, 65, 206, 97, 106], field=GF3_5)], [1, 1, 1]),
    galois.Poly([231, 15, 32, 105, 158, 158, 97, 27, 91, 4, 49, 230, 67, 8, 127, 95, 236, 179, 201, 139], field=GF3_5): ([galois.Poly([1, 162, 16, 75], field=GF3_5), galois.Poly([1, 75, 13, 71, 42, 148, 15, 202, 157, 132, 117, 165, 6, 78, 130, 159, 179], field=GF3_5)], [1, 1]),
    galois.Poly([70, 53, 84, 36, 125, 117, 54, 170, 219, 189], field=GF3_5): ([galois.Poly([1, 122], field=GF3_5), galois.Poly([1, 0, 190], field=GF3_5), galois.Poly([1, 41, 203, 17], field=GF3_5), galois.Poly([1, 238, 55, 184], field=GF3_5)], [1, 1, 1, 1]),
    galois.Poly([44, 161, 117, 189, 93, 143, 207, 70], field=GF3_5): ([galois.Poly([1, 195, 122, 26], field=GF3_5), galois.Poly([1, 65, 162, 97, 144], field=GF3_5)], [1, 1]),
    galois.Poly([235, 104, 42, 215, 51, 0, 95, 100, 57, 39, 102, 86, 227, 146, 106], field=GF3_5): ([galois.Poly([1, 20], field=GF3_5), galois.Poly([1, 189], field=GF3_5), galois.Poly([1, 202], field=GF3_5), galois.Poly([1, 22, 128, 54], field=GF3_5), galois.Poly([1, 111, 210, 119], field=GF3_5), galois.Poly([1, 168, 7, 130, 5, 66], field=GF3_5)], [1, 1, 1, 1, 1, 1]),
    galois.Poly([62, 78, 213, 174, 236, 134, 187, 57, 122, 217, 66, 103, 103], field=GF3_5): ([galois.Poly([1, 51, 177, 60], field=GF3_5), galois.Poly([1, 52, 160, 152, 212, 82, 93, 49, 131, 239], field=GF3_5)], [1, 1]),
    galois.Poly([67, 57, 175, 121, 72, 146, 144], field=GF3_5): ([galois.Poly([1, 129, 83], field=GF3_5), galois.Poly([1, 214, 22, 37, 51], field=GF3_5)], [1, 1]),
    galois.Poly([83, 200, 212, 122, 224, 200, 239, 142], field=GF3_5): ([galois.Poly([1, 30], field=GF3_5), galois.Poly([1, 84], field=GF3_5), galois.Poly([1, 45, 26], field=GF3_5), galois.Poly([1, 92, 22, 217], field=GF3_5)], [1, 1, 1, 1]),
    galois.Poly([91, 162, 175, 118, 93, 32, 70, 61, 32, 138, 82, 29, 93, 236, 191, 89, 73, 177], field=GF3_5): ([galois.Poly([1, 170], field=GF3_5), galois.Poly([1, 239, 175], field=GF3_5), galois.Poly([1, 119, 141, 182], field=GF3_5), galois.Poly([1, 227, 198, 197, 112, 140, 242, 163, 9, 125, 112, 49], field=GF3_5)], [1, 1, 1, 1]),
    galois.Poly([164, 172, 165, 112], field=GF3_5): ([galois.Poly([1, 203], field=GF3_5), galois.Poly([1, 140, 158], field=GF3_5)], [1, 1]),
    galois.Poly([6, 57, 223, 103, 20, 177, 188, 85, 134, 131, 76, 165, 126, 202, 228, 163], field=GF3_5): ([galois.Poly([1, 220], field=GF3_5), galois.Poly([1, 229], field=GF3_5), galois.Poly([1, 167, 82], field=GF3_5), galois.Poly([1, 55, 30, 116, 167, 210, 163, 21, 146, 130, 151, 10], field=GF3_5)], [1, 1, 1, 1]),
    galois.Poly([207, 210, 184, 66, 185, 188, 43, 202, 89, 231, 160, 190, 44, 106, 81, 206, 118, 122, 172], field=GF3_5): ([galois.Poly([1, 154, 165, 118, 207, 9, 39, 203, 193, 234, 109, 180, 54, 41, 37, 47, 121, 208, 226], field=GF3_5)], [1]),
    galois.Poly([169, 168, 191, 137], field=GF3_5): ([galois.Poly([1, 121], field=GF3_5), galois.Poly([1, 150, 50], field=GF3_5)], [1, 1]),
    galois.Poly([62, 196, 232, 113, 2, 23, 238, 66, 54, 110, 71, 225, 181], field=GF3_5): ([galois.Poly([1, 40], field=GF3_5), galois.Poly([1, 18, 159, 171, 56], field=GF3_5), galois.Poly([1, 195, 232, 63, 157, 50, 64, 118], field=GF3_5)], [1, 1, 1]),
    galois.Poly([219, 195, 190, 66, 64, 79, 225, 228, 32, 191, 131, 88, 75, 122, 109, 56], field=GF3_5): ([galois.Poly([1, 163], field=GF3_5), galois.Poly([1, 185], field=GF3_5), galois.Poly([1, 66, 23], field=GF3_5), galois.Poly([1, 115, 112, 223], field=GF3_5), galois.Poly([1, 237, 71, 99], field=GF3_5), galois.Poly([1, 176, 76, 28, 77, 57], field=GF3_5)], [1, 1, 1, 1, 1, 1]),
    galois.Poly([51, 152, 149, 35, 204, 173, 220, 9, 44, 205, 112, 77, 60, 86, 38, 184, 136, 178, 93], field=GF3_5): ([galois.Poly([1, 24], field=GF3_5), galois.Poly([1, 84, 67, 52, 223, 67, 118, 120, 123, 75, 27, 135, 160, 190, 132, 234, 169, 155], field=GF3_5)], [1, 1]),
    galois.Poly([177, 235, 133, 174], field=GF3_5): ([galois.Poly([1, 63], field=GF3_5), galois.Poly([1, 202], field=GF3_5), galois.Poly([1, 221], field=GF3_5)], [1, 1, 1]),
    galois.Poly([77, 236, 233, 116, 145, 23, 89, 49, 0, 242, 101, 241, 223, 231, 153, 76, 19, 185, 141, 105, 150], field=GF3_5): ([galois.Poly([1, 209], field=GF3_5), galois.Poly([1, 14, 177], field=GF3_5), galois.Poly([1, 19, 194, 184, 92, 4, 124, 82, 73, 22, 181, 232, 136, 142, 188, 240, 103, 45], field=GF3_5)], [1, 1, 1]),
    galois.Poly([46, 168], field=GF3_5): ([galois.Poly([1, 123], field=GF3_5)], [1]),
    galois.Poly([169, 165, 64, 227], field=GF3_5): ([galois.Poly([1, 242, 220, 169], field=GF3_5)], [1]),
}

POLY_FACTORS_5_1 = {
    galois.Poly([3, 4, 2, 1, 1, 3, 2, 1, 2, 0, 3, 3, 2, 2, 2], field=GF5_1): ([galois.Poly([1, 0, 4, 2], field=GF5_1), galois.Poly([1, 3, 3, 2, 2], field=GF5_1), galois.Poly([1, 0, 2, 0, 3, 1, 3, 1], field=GF5_1)], [1, 1, 1]),
    galois.Poly([3, 2, 2, 1, 3, 2, 4, 3, 4, 2, 4, 0, 4, 3, 0], field=GF5_1): ([galois.Poly([1, 0], field=GF5_1), galois.Poly([1, 3, 3, 2, 4, 4], field=GF5_1), galois.Poly([1, 1, 3, 3, 2, 0, 0, 3, 4], field=GF5_1)], [1, 1, 1]),
    galois.Poly([3, 3, 0, 4, 4, 1, 1, 4, 0, 1, 2, 1, 1, 4, 3, 2, 0, 3, 3, 2], field=GF5_1): ([galois.Poly([1, 4, 4, 4, 4], field=GF5_1), galois.Poly([1, 4, 1, 4, 1, 1], field=GF5_1), galois.Poly([1, 3, 0, 2, 2, 0, 3, 3, 4, 2, 1], field=GF5_1)], [1, 1, 1]),
    galois.Poly([3, 3, 4, 3, 3, 4, 1, 1, 1, 2, 4], field=GF5_1): ([galois.Poly([1, 0, 2], field=GF5_1), galois.Poly([1, 1, 4, 2, 1, 1, 2], field=GF5_1)], [2, 1]),
    galois.Poly([3, 0, 1, 3], field=GF5_1): ([galois.Poly([1, 0, 2, 1], field=GF5_1)], [1]),
    galois.Poly([3, 4, 1, 1, 3, 1, 1, 4, 3, 1, 3, 1], field=GF5_1): ([galois.Poly([1, 1, 4, 1, 0, 2], field=GF5_1), galois.Poly([1, 2, 1, 2, 3, 3, 1], field=GF5_1)], [1, 1]),
    galois.Poly([3, 0, 4, 2, 2, 1, 1, 1, 2, 2, 4, 3], field=GF5_1): ([galois.Poly([1, 4], field=GF5_1), galois.Poly([1, 3, 1, 3, 4, 1], field=GF5_1), galois.Poly([1, 3, 4, 0, 0, 4], field=GF5_1)], [1, 1, 1]),
    galois.Poly([3, 4, 4, 2, 2, 4, 3, 3], field=GF5_1): ([galois.Poly([1, 4], field=GF5_1), galois.Poly([1, 2, 0, 1], field=GF5_1)], [4, 1]),
    galois.Poly([3, 3, 1, 3, 3, 4, 2, 2, 1, 4, 3, 2], field=GF5_1): ([galois.Poly([1, 1], field=GF5_1), galois.Poly([1, 4, 3, 1, 1, 0, 3, 3, 3, 4], field=GF5_1)], [2, 1]),
    galois.Poly([2, 4, 3, 3, 4, 4, 4, 3, 3, 4, 4], field=GF5_1): ([galois.Poly([1, 2], field=GF5_1), galois.Poly([1, 1, 1, 1, 4], field=GF5_1), galois.Poly([1, 2, 0, 2, 2], field=GF5_1)], [2, 1, 1]),
    galois.Poly([3, 3, 2, 3, 2, 0, 2, 3, 4, 4], field=GF5_1): ([galois.Poly([1, 1], field=GF5_1), galois.Poly([1, 1, 3, 4], field=GF5_1), galois.Poly([1, 4, 2, 4, 1, 2], field=GF5_1)], [1, 1, 1]),
    galois.Poly([4, 1, 2, 4, 3, 2, 2, 3, 1, 4, 2], field=GF5_1): ([galois.Poly([1, 1], field=GF5_1), galois.Poly([1, 4, 2], field=GF5_1), galois.Poly([1, 3, 0, 4], field=GF5_1), galois.Poly([1, 1, 4, 4, 1], field=GF5_1)], [1, 1, 1, 1]),
    galois.Poly([4, 3, 3, 2, 2, 1, 3, 0, 2, 3], field=GF5_1): ([galois.Poly([1, 1], field=GF5_1), galois.Poly([1, 2, 3], field=GF5_1), galois.Poly([1, 4, 0, 0, 1, 1, 4], field=GF5_1)], [1, 1, 1]),
    galois.Poly([3, 1, 3, 3, 4, 4, 2, 0, 4, 1, 2, 3, 4, 2, 2, 3, 4, 4], field=GF5_1): ([galois.Poly([1, 2, 1, 0, 2], field=GF5_1), galois.Poly([1, 2, 1, 1, 0, 3], field=GF5_1), galois.Poly([1, 3, 3, 1, 1, 0, 1, 3, 3], field=GF5_1)], [1, 1, 1]),
    galois.Poly([2, 2, 0, 4, 0], field=GF5_1): ([galois.Poly([1, 0], field=GF5_1), galois.Poly([1, 1, 0, 2], field=GF5_1)], [1, 1]),
    galois.Poly([4, 4, 4, 0, 4], field=GF5_1): ([galois.Poly([1, 1, 1, 0, 1], field=GF5_1)], [1]),
    galois.Poly([1, 0, 1, 1, 4, 4, 3], field=GF5_1): ([galois.Poly([1, 3], field=GF5_1), galois.Poly([1, 2, 0, 1, 1, 1], field=GF5_1)], [1, 1]),
    galois.Poly([1, 2, 1, 4, 3, 1, 0, 1, 2, 4, 3], field=GF5_1): ([galois.Poly([1, 1, 1], field=GF5_1), galois.Poly([1, 1, 4, 4, 0, 2, 3, 1, 3], field=GF5_1)], [1, 1]),
    galois.Poly([3, 2, 0, 1, 3, 1, 4, 3, 1, 2, 1, 0, 4, 2, 0, 4, 1, 4, 4, 0], field=GF5_1): ([galois.Poly([1, 0], field=GF5_1), galois.Poly([1, 4], field=GF5_1), galois.Poly([1, 0, 0, 2, 3, 0, 3, 4, 1, 0, 2, 2, 0, 4, 4, 2, 4, 2], field=GF5_1)], [1, 1, 1]),
    galois.Poly([3, 1, 2, 0, 3, 4, 0, 4, 3], field=GF5_1): ([galois.Poly([1, 3], field=GF5_1), galois.Poly([1, 4], field=GF5_1), galois.Poly([1, 0, 2, 1, 0, 1, 3], field=GF5_1)], [1, 1, 1]),
}

POLY_FACTORS_5_4 = {
    galois.Poly([320, 128, 214, 469, 404, 44, 402, 7, 299, 188, 91, 597, 495, 395, 36, 238, 105, 413, 274, 386, 189], field=GF5_4): ([galois.Poly([1, 8, 427, 312], field=GF5_4), galois.Poly([1, 294, 216, 415, 369], field=GF5_4), galois.Poly([1, 541, 336, 362, 573, 577], field=GF5_4), galois.Poly([1, 541, 135, 583, 236, 318, 515, 566, 231], field=GF5_4)], [1, 1, 1, 1]),
    galois.Poly([204, 153, 544, 445, 555, 397, 546, 329, 557, 18, 543, 252, 77, 188, 477, 448, 228, 404, 189, 355, 344], field=GF5_4): ([galois.Poly([1, 517], field=GF5_4), galois.Poly([1, 606], field=GF5_4), galois.Poly([1, 449, 395], field=GF5_4), galois.Poly([1, 339, 138, 289], field=GF5_4), galois.Poly([1, 616, 454, 579, 438], field=GF5_4), galois.Poly([1, 432, 90, 514, 292, 264, 103, 552, 55, 481], field=GF5_4)], [1, 1, 1, 1, 1, 1]),
    galois.Poly([277, 623, 67, 242, 284], field=GF5_4): ([galois.Poly([1, 342], field=GF5_4), galois.Poly([1, 591], field=GF5_4), galois.Poly([1, 593, 607], field=GF5_4)], [1, 1, 1]),
    galois.Poly([181, 391, 504, 581, 572, 619], field=GF5_4): ([galois.Poly([1, 176, 504], field=GF5_4), galois.Poly([1, 591, 615, 557], field=GF5_4)], [1, 1]),
    galois.Poly([507, 3, 464, 191, 51, 515, 586, 425, 210, 296, 207, 504, 279, 423], field=GF5_4): ([galois.Poly([1, 48, 226, 138], field=GF5_4), galois.Poly([1, 198, 368, 210, 141, 163, 222, 431, 493, 416, 549], field=GF5_4)], [1, 1]),
    galois.Poly([421, 535, 552, 414, 457, 538, 30, 25, 612, 416, 166, 250, 342, 585, 532, 258, 582, 65, 177, 239], field=GF5_4): ([galois.Poly([1, 12], field=GF5_4), galois.Poly([1, 292, 526], field=GF5_4), galois.Poly([1, 608, 188, 590, 292, 577, 96, 610], field=GF5_4), galois.Poly([1, 25, 11, 239, 358, 14, 516, 479, 13, 84], field=GF5_4)], [1, 1, 1, 1]),
    galois.Poly([442, 317, 329, 25, 515, 599, 232, 500, 525, 326, 319, 450, 25, 245, 564, 203], field=GF5_4): ([galois.Poly([1, 113], field=GF5_4), galois.Poly([1, 43, 142], field=GF5_4), galois.Poly([1, 416, 347, 546], field=GF5_4), galois.Poly([1, 540, 514, 75, 254], field=GF5_4), galois.Poly([1, 482, 120, 507, 613, 607], field=GF5_4)], [1, 1, 1, 1, 1]),
    galois.Poly([385], field=GF5_4): ([], []),
    galois.Poly([486, 223, 334, 151, 14], field=GF5_4): ([galois.Poly([1, 134, 300], field=GF5_4), galois.Poly([1, 541, 198], field=GF5_4)], [1, 1]),
    galois.Poly([242, 311, 35, 236, 237, 501, 35, 471, 275], field=GF5_4): ([galois.Poly([1, 194], field=GF5_4), galois.Poly([1, 470], field=GF5_4), galois.Poly([1, 570, 111, 272, 246, 357, 32], field=GF5_4)], [1, 1, 1]),
    galois.Poly([605, 223, 450, 264, 385], field=GF5_4): ([galois.Poly([1, 292, 171, 406, 504], field=GF5_4)], [1]),
    galois.Poly([512], field=GF5_4): ([], []),
    galois.Poly([128], field=GF5_4): ([], []),
    galois.Poly([153, 338, 537, 481, 218, 78, 90, 578, 158, 202], field=GF5_4): ([galois.Poly([1, 597], field=GF5_4), galois.Poly([1, 31, 78, 470, 272, 456, 270, 345, 134], field=GF5_4)], [1, 1]),
    galois.Poly([13, 235, 614, 581, 192, 279, 166, 474, 199, 140, 335, 2], field=GF5_4): ([galois.Poly([1, 47], field=GF5_4), galois.Poly([1, 490, 597, 275, 327], field=GF5_4), galois.Poly([1, 190, 230, 615, 381, 584, 6], field=GF5_4)], [1, 1, 1]),
    galois.Poly([318, 393, 594, 318, 487, 435, 107, 66, 480, 444, 578, 508, 222, 321, 264, 372], field=GF5_4): ([galois.Poly([1, 344, 503, 1, 521, 190, 50, 25, 79, 54, 367, 586, 425, 544, 378, 254], field=GF5_4)], [1]),
    galois.Poly([337, 110, 444, 12], field=GF5_4): ([galois.Poly([1, 91], field=GF5_4), galois.Poly([1, 289, 180], field=GF5_4)], [1, 1]),
    galois.Poly([437, 535, 493, 261, 4, 338, 492, 144, 457, 569, 499, 581, 137, 323, 68, 542, 588, 43, 394, 611], field=GF5_4): ([galois.Poly([1, 318], field=GF5_4), galois.Poly([1, 278, 490, 4], field=GF5_4), galois.Poly([1, 279, 287, 503], field=GF5_4), galois.Poly([1, 100, 613, 357, 15, 544, 583], field=GF5_4), galois.Poly([1, 280, 172, 232, 263, 5, 615], field=GF5_4)], [1, 1, 1, 1, 1]),
    galois.Poly([27, 218, 424, 571, 360, 88, 125], field=GF5_4): ([galois.Poly([1, 237], field=GF5_4), galois.Poly([1, 146, 567], field=GF5_4), galois.Poly([1, 427, 560, 386], field=GF5_4)], [1, 1, 1]),
    galois.Poly([392, 566, 341, 423, 261, 620, 72, 576, 113, 506, 551, 52, 500, 481, 355, 15, 601, 595], field=GF5_4): ([galois.Poly([1, 263, 575], field=GF5_4), galois.Poly([1, 345, 199, 261], field=GF5_4), galois.Poly([1, 25, 79, 563, 258], field=GF5_4), galois.Poly([1, 30, 477, 107, 564, 64, 508, 579, 478], field=GF5_4)], [1, 1, 1, 1]),
}


def test_poly_factors_exceptions():
    GF = galois.GF(31)
    f = galois.Poly.Random(10, field=GF)

    with pytest.raises(TypeError):
        galois.poly_pow(f.coeffs)


def test_poly_factors_old():
    g0, g1, g2 = galois.conway_poly(2, 3), galois.conway_poly(2, 4), galois.conway_poly(2, 5)
    k0, k1, k2 = 2, 3, 4
    f = g0**k0 * g1**k1 * g2**k2
    factors, multiplicities = galois.poly_factors(f)
    assert factors == [g0, g1, g2]
    assert multiplicities == [k0, k1, k2]

    g0, g1, g2 = galois.conway_poly(3, 3), galois.conway_poly(3, 4), galois.conway_poly(3, 5)
    g0, g1, g2
    k0, k1, k2 = 3, 4, 6
    f = g0**k0 * g1**k1 * g2**k2
    factors, multiplicities = galois.poly_factors(f)
    assert factors == [g0, g1, g2]
    assert multiplicities == [k0, k1, k2]



# TODO: There is a bug in poly_factors() which is why this test won't pass
# @pytest.mark.parametrize("characteristic,degree", [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)])
# def test_poly_factors(characteristic, degree):
#     LUT = eval(f"POLY_FACTORS_{characteristic}_{degree}")

#     for key in LUT:
#         a = key
#         factors, multiplicities = LUT[key]

#         # Sort the Sage output to be ordered similarly to `galois`
#         factors, multiplicities = zip(*sorted(zip(factors, multiplicities), key=lambda item: item[0].integer))
#         factors, multiplicities = list(factors), list(multiplicities)

#         assert galois.poly_factors(a) == (factors, multiplicities)
