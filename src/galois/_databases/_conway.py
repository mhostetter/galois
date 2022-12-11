"""
A module that handles accessing the database of Conway polynomials.
"""
import os
import sqlite3

DATABASE = None  # Database singleton class
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "conway_polys.db")


class ConwayPolyDatabase:
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
            raise LookupError(
                f"Frank Luebeck's database of Conway polynomials doesn't contain an entry for GF({characteristic}^{degree}). "
                "See http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html for his complete list of polynomials.\n\n"
                "Alternatively, you can construct irreducible polynomials with `galois.irreducible_poly(p, m)` "
                "or primitive polynomials with `galois.primitive_poly(p, m)`."
            )

        coeffs = result[0]
        coeffs = list(map(int, coeffs[1:-1].split(",")))  # List of degree-ascending coefficients

        return coeffs[::-1]
