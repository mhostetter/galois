"""
A module containing functions to find and test Conway polynomials.
"""
from __future__ import annotations

from .._databases import ConwayPolyDatabase
from .._domains import _factory
from .._helper import export, verify_isinstance
from .._prime import is_prime
from ._poly import Poly


@export
def conway_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns the Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Arguments:
        characteristic: The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial
            is over.
        degree: The degree :math:`m` of the Conway polynomial.

    Returns:
        The degree-:math:`m` Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)`.

    See Also:
        Poly.is_primitive, primitive_poly, matlab_primitive_poly

    Raises:
        LookupError: If the Conway polynomial :math:`C_{p,m}(x)` is not found in Frank Luebeck's database.

    Notes:
        A Conway polynomial is an irreducible and primitive polynomial over :math:`\mathrm{GF}(p)` that provides a
        standard representation of :math:`\mathrm{GF}(p^m)` as a splitting field of :math:`C_{p,m}(x)`.
        Conway polynomials provide compatibility between fields and their subfields and, hence, are the common way to
        represent extension fields.

        The Conway polynomial :math:`C_{p,m}(x)` is defined as the lexicographically-first monic primitive polynomial
        of degree :math:`m` over :math:`\mathrm{GF}(p)` that is compatible with all :math:`C_{p,n}(x)` for :math:`n`
        dividing :math:`m`.

        This function uses `Frank Luebeck's Conway polynomial database
        <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_ for fast lookup, not construction.

    Examples:
        Notice :func:`~galois.primitive_poly` returns the lexicographically-first primitive polynomial but
        :func:`~galois.conway_poly` returns the lexicographically-first primitive polynomial that is *consistent*
        with smaller Conway polynomials. This is sometimes the same polynomial.

        .. ipython:: python

            galois.primitive_poly(2, 4)
            galois.conway_poly(2, 4)

        However, it is not always.

        .. ipython:: python

            galois.primitive_poly(7, 10)
            galois.conway_poly(7, 10)

    Group:
        polys-primitive
    """
    verify_isinstance(characteristic, int)
    verify_isinstance(degree, int)

    if not is_prime(characteristic):
        raise ValueError(f"Argument 'characteristic' must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(
            f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0."
        )

    db = ConwayPolyDatabase()
    coeffs = db.fetch(characteristic, degree)
    field = _factory.FIELD_FACTORY(characteristic)
    poly = Poly(coeffs, field=field)

    return poly
