import os
import sqlite3

import numpy as np

from ..overrides import set_module
from ..prime import is_prime

from .factory_prime import GF_prime
from .poly import Poly

__all__ = ["conway_poly"]

DATABASE = None  # Database singleton class
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "..", "databases", "conway_polys.db")


class ConwayDatabase:
    """
    Class to interface with the Conway polynomials database.
    """

    def __new__(cls):
        global DATABASE
        if DATABASE is None:
            DATABASE = super().__new__(cls)
        return DATABASE

    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()

    def fetch(self, characteristic, degree):
        self.cursor.execute("SELECT coefficients FROM polys WHERE characteristic=? AND degree=?", (int(characteristic), int(degree)))
        result = self.cursor.fetchone()

        if result is None:
            raise LookupError(f"Frank Luebeck's database of Conway polynomials doesn't contain an entry for GF({characteristic}^{degree}). See here http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html for his complete list of polynomials.")

        coeffs = result[0]
        coeffs = list(map(int, coeffs[1:-1].split(",")))  # List of degree-ascending coefficients

        return coeffs[::-1]


@set_module("galois")
def conway_poly(p, n):
    """
    Returns the degree-:math:`n` Conway polynomial :math:`C_{p,n}` over :math:`\\mathrm{GF}(p)`.

    A Conway polynomial is a an irreducible and primitive polynomial over :math:`\\mathrm{GF}(p)` that provides a standard
    representation of :math:`\\mathrm{GF}(p^n)` as a splitting field of :math:`C_{p,n}`. Conway polynomials
    provide compatability between fields and their subfields, and hence are the common way to represent extension
    fields.

    The Conway polynomial :math:`C_{p,n}` is defined as the lexicographically-minimal monic irreducible polynomial
    of degree :math:`n` over :math:`\\mathrm{GF}(p)` that is compatible with all :math:`C_{p,m}` for :math:`m` dividing
    :math:`n`.

    This function uses Frank Luebeck's Conway polynomial database for fast lookup, not construction.

    Parameters
    ----------
    p : int
        The prime characteristic of the field :math:`\\mathrm{GF}(p)`.
    n : int
        The degree :math:`n` of the Conway polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`n` Conway polynomial :math:`C_{p,n}` over :math:`\\mathrm{GF}(p)`.

    Raises
    ------
    LookupError
        If the Conway polynomial :math:`C_{p,n}` is not found in Frank Luebeck's database.

    Warning
    -------
    If the :math:`\\mathrm{GF}(p)` field hasn't previously been created, it will be created in this function
    since it's needed for the construction of the return polynomial.

    Examples
    --------
    .. ipython:: python

        galois.conway_poly(2, 100)
        galois.conway_poly(7, 13)
    """
    if not isinstance(p, (int, np.integer)):
        raise TypeError(f"Argument `p` must be an integer, not {type(p)}")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}")
    if not is_prime(p):
        raise ValueError(f"Argument `p` must be prime, not {p}")
    if not n >= 1:
        raise ValueError(f"Argument `n` must be at least 1, not {n}")

    coeffs = ConwayDatabase().fetch(p, n)
    field = GF_prime(p)
    poly = Poly(coeffs, field=field)

    return poly
