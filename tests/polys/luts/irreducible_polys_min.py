"""
A module containing LUTs for lexicographically first irreducible polynomials with minimal terms.

LUT items obtained by randomly picking degrees and checking the PDF,

References:
    - Gadiel Seroussi's table (1998). https://www.hpl.hp.com/techreports/98/HPL-98-135.pdf.
"""

# LUT items are (order, degree, nonzero_degrees, nonzero_coeffs) in degree-descending order.
# sorted(numpy.random.default_rng(1337).integers(size=5, low=500, high=10_000, endpoint=True))

IRREDUCIBLE_POLYS_MIN = [
    (2, 2262, [[2262, 57, 0], [1, 1, 1]]),
    (2, 5632, [[5632, 17, 15, 5, 0], [1, 1, 1, 1, 1]]),
    (2, 5690, [[5690, 1623, 0], [1, 1, 1]]),
    (2, 7407, [[7407, 27, 21, 17, 0], [1, 1, 1, 1, 1]]),
    (2, 8842, [[8842, 4143, 0], [1, 1, 1]]),
]
