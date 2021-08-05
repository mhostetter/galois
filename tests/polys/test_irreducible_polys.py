"""
A pytest module to test generating irreducible polynomials and testing irreducibility.

Sage:
    p = 3
    for degree in range(1, 7):
        print(f"IRREDUCIBLE_POLYS_{p}_{degree} = [")
        F = GF(p)["x"]
        for f in F.polynomials(degree):
            if f.is_monic() and f.is_irreducible():
                c = f.coefficients(sparse=False)[::-1]
                print(f"    galois.Poly({c}, field=GF{p}),")
        print("]\n")
"""
import pytest

import galois

GF2 = galois.GF(2)
GF3 = galois.GF(3)
GF5 = galois.GF(5)

IRREDUCIBLE_POLYS_2_1 = [
    galois.Poly([1, 0], field=GF2),
    galois.Poly([1, 1], field=GF2),
]

IRREDUCIBLE_POLYS_2_2 = [
    galois.Poly([1, 1, 1], field=GF2),
]

IRREDUCIBLE_POLYS_2_3 = [
    galois.Poly([1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1], field=GF2),
]

IRREDUCIBLE_POLYS_2_4 = [
    galois.Poly([1, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 1], field=GF2),
]

IRREDUCIBLE_POLYS_2_5 = [
    galois.Poly([1, 0, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 1], field=GF2),
]

IRREDUCIBLE_POLYS_2_6 = [
    galois.Poly([1, 0, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 0, 1], field=GF2),
]

IRREDUCIBLE_POLYS_2_7 = [
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

IRREDUCIBLE_POLYS_2_8 = [
    galois.Poly([1, 0, 0, 0, 1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 0, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 1, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 1, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 1, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 1, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 0, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 1, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 1, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 1, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 1, 1, 0, 0, 1], field=GF2),
]

IRREDUCIBLE_POLYS_3_1 = [
    galois.Poly([1, 0], field=GF3),
    galois.Poly([1, 1], field=GF3),
    galois.Poly([1, 2], field=GF3),
]

IRREDUCIBLE_POLYS_3_2 = [
    galois.Poly([1, 0, 1], field=GF3),
    galois.Poly([1, 1, 2], field=GF3),
    galois.Poly([1, 2, 2], field=GF3),
]

IRREDUCIBLE_POLYS_3_3 = [
    galois.Poly([1, 0, 2, 1], field=GF3),
    galois.Poly([1, 0, 2, 2], field=GF3),
    galois.Poly([1, 1, 0, 2], field=GF3),
    galois.Poly([1, 1, 1, 2], field=GF3),
    galois.Poly([1, 1, 2, 1], field=GF3),
    galois.Poly([1, 2, 0, 1], field=GF3),
    galois.Poly([1, 2, 1, 1], field=GF3),
    galois.Poly([1, 2, 2, 2], field=GF3),
]

IRREDUCIBLE_POLYS_3_4 = [
    galois.Poly([1, 0, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 0, 2], field=GF3),
    galois.Poly([1, 0, 1, 1, 1], field=GF3),
    galois.Poly([1, 0, 1, 2, 1], field=GF3),
    galois.Poly([1, 0, 2, 0, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 0, 2, 1], field=GF3),
    galois.Poly([1, 1, 1, 0, 1], field=GF3),
    galois.Poly([1, 1, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 2, 2], field=GF3),
    galois.Poly([1, 2, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 1, 1], field=GF3),
    galois.Poly([1, 2, 1, 0, 1], field=GF3),
    galois.Poly([1, 2, 1, 1, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 1], field=GF3),
    galois.Poly([1, 2, 2, 1, 2], field=GF3),
]

IRREDUCIBLE_POLYS_3_5 = [
    galois.Poly([1, 0, 0, 0, 2, 1], field=GF3),
    galois.Poly([1, 0, 0, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 0, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 2, 1, 1], field=GF3),
    galois.Poly([1, 0, 1, 0, 1, 1], field=GF3),
    galois.Poly([1, 0, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 1, 1, 0, 2], field=GF3),
    galois.Poly([1, 0, 1, 1, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 2, 0, 1], field=GF3),
    galois.Poly([1, 0, 1, 2, 2, 1], field=GF3),
    galois.Poly([1, 0, 2, 1, 0, 1], field=GF3),
    galois.Poly([1, 0, 2, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 0, 2, 2, 0, 2], field=GF3),
    galois.Poly([1, 0, 2, 2, 1, 1], field=GF3),
    galois.Poly([1, 0, 2, 2, 2, 1], field=GF3),
    galois.Poly([1, 1, 0, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 1, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 2, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 0, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 1, 0, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 1, 2, 1], field=GF3),
    galois.Poly([1, 1, 1, 2, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 2, 1, 2], field=GF3),
    galois.Poly([1, 1, 2, 0, 0, 1], field=GF3),
    galois.Poly([1, 1, 2, 0, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 0, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 2, 2, 0, 1], field=GF3),
    galois.Poly([1, 1, 2, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 0, 0, 1], field=GF3),
    galois.Poly([1, 2, 0, 0, 1, 1], field=GF3),
    galois.Poly([1, 2, 0, 0, 2, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 1, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 2, 1], field=GF3),
    galois.Poly([1, 2, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 2, 1, 1, 1, 1], field=GF3),
    galois.Poly([1, 2, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 2, 2], field=GF3),
    galois.Poly([1, 2, 2, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 0, 2, 1], field=GF3),
    galois.Poly([1, 2, 2, 1, 0, 1], field=GF3),
    galois.Poly([1, 2, 2, 1, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 2, 0, 1], field=GF3),
    galois.Poly([1, 2, 2, 2, 1, 2], field=GF3),
]

IRREDUCIBLE_POLYS_3_6 = [
    galois.Poly([1, 0, 0, 0, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 0, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 0, 0, 1, 1, 1], field=GF3),
    galois.Poly([1, 0, 0, 0, 1, 2, 1], field=GF3),
    galois.Poly([1, 0, 0, 0, 2, 0, 1], field=GF3),
    galois.Poly([1, 0, 0, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 1, 0, 2, 1], field=GF3),
    galois.Poly([1, 0, 0, 1, 1, 0, 1], field=GF3),
    galois.Poly([1, 0, 0, 1, 1, 2, 2], field=GF3),
    galois.Poly([1, 0, 0, 1, 2, 2, 1], field=GF3),
    galois.Poly([1, 0, 0, 2, 0, 1, 1], field=GF3),
    galois.Poly([1, 0, 0, 2, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 0, 2, 1, 0, 1], field=GF3),
    galois.Poly([1, 0, 0, 2, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 0, 2, 2, 1, 1], field=GF3),
    galois.Poly([1, 0, 1, 0, 2, 0, 1], field=GF3),
    galois.Poly([1, 0, 1, 0, 2, 1, 2], field=GF3),
    galois.Poly([1, 0, 1, 0, 2, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 1, 0, 0, 1], field=GF3),
    galois.Poly([1, 0, 1, 1, 0, 1, 1], field=GF3),
    galois.Poly([1, 0, 1, 1, 0, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 1, 1, 2, 2], field=GF3),
    galois.Poly([1, 0, 1, 2, 0, 0, 1], field=GF3),
    galois.Poly([1, 0, 1, 2, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 1, 2, 0, 2, 1], field=GF3),
    galois.Poly([1, 0, 1, 2, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 0, 0, 0, 1], field=GF3),
    galois.Poly([1, 0, 2, 0, 1, 0, 1], field=GF3),
    galois.Poly([1, 0, 2, 0, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 0, 1, 2, 2], field=GF3),
    galois.Poly([1, 0, 2, 1, 0, 2, 1], field=GF3),
    galois.Poly([1, 0, 2, 1, 1, 0, 2], field=GF3),
    galois.Poly([1, 0, 2, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 1, 1, 2, 1], field=GF3),
    galois.Poly([1, 0, 2, 2, 0, 1, 1], field=GF3),
    galois.Poly([1, 0, 2, 2, 1, 0, 2], field=GF3),
    galois.Poly([1, 0, 2, 2, 1, 1, 1], field=GF3),
    galois.Poly([1, 0, 2, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 0, 1, 2], field=GF3),
    galois.Poly([1, 1, 0, 0, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 0, 1, 0, 1, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 1, 0, 1], field=GF3),
    galois.Poly([1, 1, 0, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 1, 0, 1, 2, 1, 2], field=GF3),
    galois.Poly([1, 1, 0, 2, 0, 0, 1], field=GF3),
    galois.Poly([1, 1, 0, 2, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 0, 2, 1, 2, 1], field=GF3),
    galois.Poly([1, 1, 0, 2, 2, 0, 1], field=GF3),
    galois.Poly([1, 1, 0, 2, 2, 0, 2], field=GF3),
    galois.Poly([1, 1, 1, 0, 0, 0, 1], field=GF3),
    galois.Poly([1, 1, 1, 0, 0, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 0, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 1, 0, 2, 0, 2], field=GF3),
    galois.Poly([1, 1, 1, 0, 2, 2, 1], field=GF3),
    galois.Poly([1, 1, 1, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 1, 1, 1, 0, 2, 1], field=GF3),
    galois.Poly([1, 1, 1, 1, 1, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 1, 1, 1, 2, 2, 2], field=GF3),
    galois.Poly([1, 1, 1, 2, 0, 1, 1], field=GF3),
    galois.Poly([1, 1, 1, 2, 2, 0, 1], field=GF3),
    galois.Poly([1, 1, 1, 2, 2, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 0, 1, 0, 2], field=GF3),
    galois.Poly([1, 1, 2, 0, 1, 2, 1], field=GF3),
    galois.Poly([1, 1, 2, 0, 2, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 0, 1, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 1, 0, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 2, 1, 2], field=GF3),
    galois.Poly([1, 1, 2, 1, 2, 2, 1], field=GF3),
    galois.Poly([1, 1, 2, 2, 0, 0, 1], field=GF3),
    galois.Poly([1, 1, 2, 2, 0, 0, 2], field=GF3),
    galois.Poly([1, 1, 2, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 2, 2, 0, 2], field=GF3),
    galois.Poly([1, 1, 2, 2, 2, 2, 1], field=GF3),
    galois.Poly([1, 2, 0, 0, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 0, 0, 2, 2], field=GF3),
    galois.Poly([1, 2, 0, 0, 1, 2, 1], field=GF3),
    galois.Poly([1, 2, 0, 1, 0, 0, 1], field=GF3),
    galois.Poly([1, 2, 0, 1, 1, 1, 1], field=GF3),
    galois.Poly([1, 2, 0, 1, 1, 2, 1], field=GF3),
    galois.Poly([1, 2, 0, 1, 2, 0, 1], field=GF3),
    galois.Poly([1, 2, 0, 1, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 0, 2, 1], field=GF3),
    galois.Poly([1, 2, 0, 2, 1, 0, 1], field=GF3),
    galois.Poly([1, 2, 0, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 2, 0, 2, 2, 2, 2], field=GF3),
    galois.Poly([1, 2, 1, 0, 0, 0, 1], field=GF3),
    galois.Poly([1, 2, 1, 0, 0, 2, 1], field=GF3),
    galois.Poly([1, 2, 1, 0, 1, 1, 2], field=GF3),
    galois.Poly([1, 2, 1, 0, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 1, 0, 2, 1, 1], field=GF3),
    galois.Poly([1, 2, 1, 1, 0, 2, 1], field=GF3),
    galois.Poly([1, 2, 1, 1, 2, 0, 1], field=GF3),
    galois.Poly([1, 2, 1, 1, 2, 1, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 0, 1, 1], field=GF3),
    galois.Poly([1, 2, 1, 2, 0, 2, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 1, 2, 1], field=GF3),
    galois.Poly([1, 2, 1, 2, 1, 2, 2], field=GF3),
    galois.Poly([1, 2, 1, 2, 2, 1, 2], field=GF3),
    galois.Poly([1, 2, 2, 0, 1, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 0, 1, 1, 1], field=GF3),
    galois.Poly([1, 2, 2, 0, 2, 1, 2], field=GF3),
    galois.Poly([1, 2, 2, 1, 0, 0, 1], field=GF3),
    galois.Poly([1, 2, 2, 1, 0, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 1, 1, 1, 2], field=GF3),
    galois.Poly([1, 2, 2, 1, 2, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 1, 2, 1, 1], field=GF3),
    galois.Poly([1, 2, 2, 2, 0, 2, 2], field=GF3),
    galois.Poly([1, 2, 2, 2, 1, 0, 2], field=GF3),
    galois.Poly([1, 2, 2, 2, 1, 1, 2], field=GF3),
    galois.Poly([1, 2, 2, 2, 2, 1, 1], field=GF3),
    galois.Poly([1, 2, 2, 2, 2, 2, 2], field=GF3),
]

IRREDUCIBLE_POLYS_5_1 = [
    galois.Poly([1, 0], field=GF5),
    galois.Poly([1, 1], field=GF5),
    galois.Poly([1, 2], field=GF5),
    galois.Poly([1, 3], field=GF5),
    galois.Poly([1, 4], field=GF5),
]

IRREDUCIBLE_POLYS_5_2 = [
    galois.Poly([1, 0, 2], field=GF5),
    galois.Poly([1, 0, 3], field=GF5),
    galois.Poly([1, 1, 1], field=GF5),
    galois.Poly([1, 1, 2], field=GF5),
    galois.Poly([1, 2, 3], field=GF5),
    galois.Poly([1, 2, 4], field=GF5),
    galois.Poly([1, 3, 3], field=GF5),
    galois.Poly([1, 3, 4], field=GF5),
    galois.Poly([1, 4, 1], field=GF5),
    galois.Poly([1, 4, 2], field=GF5),
]

IRREDUCIBLE_POLYS_5_3 = [
    galois.Poly([1, 0, 1, 1], field=GF5),
    galois.Poly([1, 0, 1, 4], field=GF5),
    galois.Poly([1, 0, 2, 1], field=GF5),
    galois.Poly([1, 0, 2, 4], field=GF5),
    galois.Poly([1, 0, 3, 2], field=GF5),
    galois.Poly([1, 0, 3, 3], field=GF5),
    galois.Poly([1, 0, 4, 2], field=GF5),
    galois.Poly([1, 0, 4, 3], field=GF5),
    galois.Poly([1, 1, 0, 1], field=GF5),
    galois.Poly([1, 1, 0, 2], field=GF5),
    galois.Poly([1, 1, 1, 3], field=GF5),
    galois.Poly([1, 1, 1, 4], field=GF5),
    galois.Poly([1, 1, 3, 1], field=GF5),
    galois.Poly([1, 1, 3, 4], field=GF5),
    galois.Poly([1, 1, 4, 1], field=GF5),
    galois.Poly([1, 1, 4, 3], field=GF5),
    galois.Poly([1, 2, 0, 1], field=GF5),
    galois.Poly([1, 2, 0, 3], field=GF5),
    galois.Poly([1, 2, 1, 3], field=GF5),
    galois.Poly([1, 2, 1, 4], field=GF5),
    galois.Poly([1, 2, 2, 2], field=GF5),
    galois.Poly([1, 2, 2, 3], field=GF5),
    galois.Poly([1, 2, 4, 2], field=GF5),
    galois.Poly([1, 2, 4, 4], field=GF5),
    galois.Poly([1, 3, 0, 2], field=GF5),
    galois.Poly([1, 3, 0, 4], field=GF5),
    galois.Poly([1, 3, 1, 1], field=GF5),
    galois.Poly([1, 3, 1, 2], field=GF5),
    galois.Poly([1, 3, 2, 2], field=GF5),
    galois.Poly([1, 3, 2, 3], field=GF5),
    galois.Poly([1, 3, 4, 1], field=GF5),
    galois.Poly([1, 3, 4, 3], field=GF5),
    galois.Poly([1, 4, 0, 3], field=GF5),
    galois.Poly([1, 4, 0, 4], field=GF5),
    galois.Poly([1, 4, 1, 1], field=GF5),
    galois.Poly([1, 4, 1, 2], field=GF5),
    galois.Poly([1, 4, 3, 1], field=GF5),
    galois.Poly([1, 4, 3, 4], field=GF5),
    galois.Poly([1, 4, 4, 2], field=GF5),
    galois.Poly([1, 4, 4, 4], field=GF5),
]

IRREDUCIBLE_POLYS_5_4 = [
    galois.Poly([1, 0, 0, 0, 2], field=GF5),
    galois.Poly([1, 0, 0, 0, 3], field=GF5),
    galois.Poly([1, 0, 0, 1, 4], field=GF5),
    galois.Poly([1, 0, 0, 2, 4], field=GF5),
    galois.Poly([1, 0, 0, 3, 4], field=GF5),
    galois.Poly([1, 0, 0, 4, 4], field=GF5),
    galois.Poly([1, 0, 1, 0, 2], field=GF5),
    galois.Poly([1, 0, 1, 1, 1], field=GF5),
    galois.Poly([1, 0, 1, 2, 2], field=GF5),
    galois.Poly([1, 0, 1, 2, 3], field=GF5),
    galois.Poly([1, 0, 1, 3, 2], field=GF5),
    galois.Poly([1, 0, 1, 3, 3], field=GF5),
    galois.Poly([1, 0, 1, 4, 1], field=GF5),
    galois.Poly([1, 0, 2, 0, 3], field=GF5),
    galois.Poly([1, 0, 2, 2, 1], field=GF5),
    galois.Poly([1, 0, 2, 2, 3], field=GF5),
    galois.Poly([1, 0, 2, 3, 1], field=GF5),
    galois.Poly([1, 0, 2, 3, 3], field=GF5),
    galois.Poly([1, 0, 3, 0, 3], field=GF5),
    galois.Poly([1, 0, 3, 1, 1], field=GF5),
    galois.Poly([1, 0, 3, 1, 3], field=GF5),
    galois.Poly([1, 0, 3, 4, 1], field=GF5),
    galois.Poly([1, 0, 3, 4, 3], field=GF5),
    galois.Poly([1, 0, 4, 0, 2], field=GF5),
    galois.Poly([1, 0, 4, 1, 2], field=GF5),
    galois.Poly([1, 0, 4, 1, 3], field=GF5),
    galois.Poly([1, 0, 4, 2, 1], field=GF5),
    galois.Poly([1, 0, 4, 3, 1], field=GF5),
    galois.Poly([1, 0, 4, 4, 2], field=GF5),
    galois.Poly([1, 0, 4, 4, 3], field=GF5),
    galois.Poly([1, 1, 0, 0, 4], field=GF5),
    galois.Poly([1, 1, 0, 1, 3], field=GF5),
    galois.Poly([1, 1, 0, 2, 3], field=GF5),
    galois.Poly([1, 1, 0, 2, 4], field=GF5),
    galois.Poly([1, 1, 0, 3, 2], field=GF5),
    galois.Poly([1, 1, 0, 4, 1], field=GF5),
    galois.Poly([1, 1, 0, 4, 2], field=GF5),
    galois.Poly([1, 1, 1, 0, 1], field=GF5),
    galois.Poly([1, 1, 1, 1, 3], field=GF5),
    galois.Poly([1, 1, 1, 1, 4], field=GF5),
    galois.Poly([1, 1, 1, 2, 4], field=GF5),
    galois.Poly([1, 1, 1, 3, 3], field=GF5),
    galois.Poly([1, 1, 1, 4, 2], field=GF5),
    galois.Poly([1, 1, 2, 0, 2], field=GF5),
    galois.Poly([1, 1, 2, 1, 2], field=GF5),
    galois.Poly([1, 1, 2, 1, 3], field=GF5),
    galois.Poly([1, 1, 2, 2, 1], field=GF5),
    galois.Poly([1, 1, 2, 2, 2], field=GF5),
    galois.Poly([1, 1, 2, 3, 4], field=GF5),
    galois.Poly([1, 1, 2, 4, 4], field=GF5),
    galois.Poly([1, 1, 3, 0, 1], field=GF5),
    galois.Poly([1, 1, 3, 0, 3], field=GF5),
    galois.Poly([1, 1, 3, 2, 1], field=GF5),
    galois.Poly([1, 1, 3, 4, 2], field=GF5),
    galois.Poly([1, 1, 3, 4, 4], field=GF5),
    galois.Poly([1, 1, 4, 0, 2], field=GF5),
    galois.Poly([1, 1, 4, 1, 1], field=GF5),
    galois.Poly([1, 1, 4, 1, 4], field=GF5),
    galois.Poly([1, 1, 4, 4, 1], field=GF5),
    galois.Poly([1, 1, 4, 4, 3], field=GF5),
    galois.Poly([1, 2, 0, 0, 4], field=GF5),
    galois.Poly([1, 2, 0, 1, 3], field=GF5),
    galois.Poly([1, 2, 0, 1, 4], field=GF5),
    galois.Poly([1, 2, 0, 2, 1], field=GF5),
    galois.Poly([1, 2, 0, 2, 2], field=GF5),
    galois.Poly([1, 2, 0, 3, 3], field=GF5),
    galois.Poly([1, 2, 0, 4, 2], field=GF5),
    galois.Poly([1, 2, 1, 0, 2], field=GF5),
    galois.Poly([1, 2, 1, 2, 1], field=GF5),
    galois.Poly([1, 2, 1, 2, 3], field=GF5),
    galois.Poly([1, 2, 1, 3, 1], field=GF5),
    galois.Poly([1, 2, 1, 3, 4], field=GF5),
    galois.Poly([1, 2, 2, 0, 1], field=GF5),
    galois.Poly([1, 2, 2, 0, 3], field=GF5),
    galois.Poly([1, 2, 2, 1, 1], field=GF5),
    galois.Poly([1, 2, 2, 2, 2], field=GF5),
    galois.Poly([1, 2, 2, 2, 4], field=GF5),
    galois.Poly([1, 2, 3, 0, 2], field=GF5),
    galois.Poly([1, 2, 3, 1, 1], field=GF5),
    galois.Poly([1, 2, 3, 1, 2], field=GF5),
    galois.Poly([1, 2, 3, 2, 4], field=GF5),
    galois.Poly([1, 2, 3, 3, 2], field=GF5),
    galois.Poly([1, 2, 3, 3, 3], field=GF5),
    galois.Poly([1, 2, 3, 4, 4], field=GF5),
    galois.Poly([1, 2, 4, 0, 1], field=GF5),
    galois.Poly([1, 2, 4, 1, 4], field=GF5),
    galois.Poly([1, 2, 4, 2, 2], field=GF5),
    galois.Poly([1, 2, 4, 3, 3], field=GF5),
    galois.Poly([1, 2, 4, 3, 4], field=GF5),
    galois.Poly([1, 2, 4, 4, 3], field=GF5),
    galois.Poly([1, 3, 0, 0, 4], field=GF5),
    galois.Poly([1, 3, 0, 1, 2], field=GF5),
    galois.Poly([1, 3, 0, 2, 3], field=GF5),
    galois.Poly([1, 3, 0, 3, 1], field=GF5),
    galois.Poly([1, 3, 0, 3, 2], field=GF5),
    galois.Poly([1, 3, 0, 4, 3], field=GF5),
    galois.Poly([1, 3, 0, 4, 4], field=GF5),
    galois.Poly([1, 3, 1, 0, 2], field=GF5),
    galois.Poly([1, 3, 1, 2, 1], field=GF5),
    galois.Poly([1, 3, 1, 2, 4], field=GF5),
    galois.Poly([1, 3, 1, 3, 1], field=GF5),
    galois.Poly([1, 3, 1, 3, 3], field=GF5),
    galois.Poly([1, 3, 2, 0, 1], field=GF5),
    galois.Poly([1, 3, 2, 0, 3], field=GF5),
    galois.Poly([1, 3, 2, 3, 2], field=GF5),
    galois.Poly([1, 3, 2, 3, 4], field=GF5),
    galois.Poly([1, 3, 2, 4, 1], field=GF5),
    galois.Poly([1, 3, 3, 0, 2], field=GF5),
    galois.Poly([1, 3, 3, 1, 4], field=GF5),
    galois.Poly([1, 3, 3, 2, 2], field=GF5),
    galois.Poly([1, 3, 3, 2, 3], field=GF5),
    galois.Poly([1, 3, 3, 3, 4], field=GF5),
    galois.Poly([1, 3, 3, 4, 1], field=GF5),
    galois.Poly([1, 3, 3, 4, 2], field=GF5),
    galois.Poly([1, 3, 4, 0, 1], field=GF5),
    galois.Poly([1, 3, 4, 1, 3], field=GF5),
    galois.Poly([1, 3, 4, 2, 3], field=GF5),
    galois.Poly([1, 3, 4, 2, 4], field=GF5),
    galois.Poly([1, 3, 4, 3, 2], field=GF5),
    galois.Poly([1, 3, 4, 4, 4], field=GF5),
    galois.Poly([1, 4, 0, 0, 4], field=GF5),
    galois.Poly([1, 4, 0, 1, 1], field=GF5),
    galois.Poly([1, 4, 0, 1, 2], field=GF5),
    galois.Poly([1, 4, 0, 2, 2], field=GF5),
    galois.Poly([1, 4, 0, 3, 3], field=GF5),
    galois.Poly([1, 4, 0, 3, 4], field=GF5),
    galois.Poly([1, 4, 0, 4, 3], field=GF5),
    galois.Poly([1, 4, 1, 0, 1], field=GF5),
    galois.Poly([1, 4, 1, 1, 2], field=GF5),
    galois.Poly([1, 4, 1, 2, 3], field=GF5),
    galois.Poly([1, 4, 1, 3, 4], field=GF5),
    galois.Poly([1, 4, 1, 4, 3], field=GF5),
    galois.Poly([1, 4, 1, 4, 4], field=GF5),
    galois.Poly([1, 4, 2, 0, 2], field=GF5),
    galois.Poly([1, 4, 2, 1, 4], field=GF5),
    galois.Poly([1, 4, 2, 2, 4], field=GF5),
    galois.Poly([1, 4, 2, 3, 1], field=GF5),
    galois.Poly([1, 4, 2, 3, 2], field=GF5),
    galois.Poly([1, 4, 2, 4, 2], field=GF5),
    galois.Poly([1, 4, 2, 4, 3], field=GF5),
    galois.Poly([1, 4, 3, 0, 1], field=GF5),
    galois.Poly([1, 4, 3, 0, 3], field=GF5),
    galois.Poly([1, 4, 3, 1, 2], field=GF5),
    galois.Poly([1, 4, 3, 1, 4], field=GF5),
    galois.Poly([1, 4, 3, 3, 1], field=GF5),
    galois.Poly([1, 4, 4, 0, 2], field=GF5),
    galois.Poly([1, 4, 4, 1, 1], field=GF5),
    galois.Poly([1, 4, 4, 1, 3], field=GF5),
    galois.Poly([1, 4, 4, 4, 1], field=GF5),
    galois.Poly([1, 4, 4, 4, 4], field=GF5),
]


def test_irreducible_poly_exceptions():
    with pytest.raises(TypeError):
        galois.irreducible_poly(2.0, 3)
    with pytest.raises(TypeError):
        galois.irreducible_poly(2, 3.0)
    with pytest.raises(ValueError):
        galois.irreducible_poly(4, 3)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 0)
    with pytest.raises(ValueError):
        galois.irreducible_poly(2, 3, method="invalid-argument")


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_irreducible_poly_min(characteristic, degree):
    LUT = eval(f"IRREDUCIBLE_POLYS_{characteristic}_{degree}")
    assert galois.irreducible_poly(characteristic, degree) == LUT[0]


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_irreducible_poly_max(characteristic, degree):
    LUT = eval(f"IRREDUCIBLE_POLYS_{characteristic}_{degree}")
    assert galois.irreducible_poly(characteristic, degree, method="max") == LUT[-1]


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_irreducible_poly_random(characteristic, degree):
    LUT = eval(f"IRREDUCIBLE_POLYS_{characteristic}_{degree}")
    assert galois.irreducible_poly(characteristic, degree, method="random") in LUT


def test_irreducible_polys_exceptions():
    with pytest.raises(TypeError):
        galois.irreducible_polys(2.0, 3)
    with pytest.raises(TypeError):
        galois.irreducible_polys(2, 3.0)
    with pytest.raises(ValueError):
        galois.irreducible_polys(4, 3)
    with pytest.raises(ValueError):
        galois.irreducible_polys(2, 0)


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_irreducible_polys(characteristic, degree):
    LUT = eval(f"IRREDUCIBLE_POLYS_{characteristic}_{degree}")
    assert galois.irreducible_polys(characteristic, degree) == LUT


def test_is_irreducible_exceptions():
    with pytest.raises(TypeError):
        galois.is_irreducible([1, 0, 1, 1])
    with pytest.raises(ValueError):
        galois.is_irreducible(galois.Poly([1]))
    with pytest.raises(ValueError):
        galois.is_irreducible(galois.Poly([1], field=galois.GF(2**2)))


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (5,1), (5,2), (5,3), (5,4)])
def test_is_irreducible(characteristic, degree):
    LUT = eval(f"IRREDUCIBLE_POLYS_{characteristic}_{degree}")
    assert all(galois.is_irreducible(f) for f in LUT)


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,2), (3,3), (3,4), (3,5), (3,6), (5,2), (5,3), (5,4)])
def test_is_not_irreducible(characteristic, degree):
    LUT = eval(f"IRREDUCIBLE_POLYS_{characteristic}_{degree}")
    while True:
        f = galois.Poly.Random(degree, field=galois.GF(characteristic))
        f /= f.coeffs[0]  # Make monic
        if f not in LUT:
            break
    assert not galois.is_irreducible(f)
