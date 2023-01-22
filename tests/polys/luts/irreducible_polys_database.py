"""
A module containing LUTs for irreducible polynomials with min terms from irreducible_polys.db
"""

# LUT items are poly nonzero degrees and coefficients in degree-descending order

# Gadiel Seroussi's table (1998)
# LUT items obtained by randomly picking degrees and checking the PDF
# sorted(numpy.random.default_rng(1337).integers(size=5, low=500, high=10_000, endpoint=True))

IRREDUCIBLE_POLY_MIN_TERMS_2_2262 = [[2262, 57, 0], [1, 1, 1]]
IRREDUCIBLE_POLY_MIN_TERMS_2_5632 = [[5632, 17, 15, 5, 0], [1, 1, 1, 1, 1]]
IRREDUCIBLE_POLY_MIN_TERMS_2_5690 = [[5690, 1623, 0], [1, 1, 1]]
IRREDUCIBLE_POLY_MIN_TERMS_2_7407 = [[7407, 27, 21, 17, 0], [1, 1, 1, 1, 1]]
IRREDUCIBLE_POLY_MIN_TERMS_2_8842 = [[8842, 4143, 0], [1, 1, 1]]
