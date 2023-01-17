"""
A module that handles accessing the database of irreducible polynomials.
"""
import os
import sqlite3

DATABASE = None  # Database singleton class
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "irreducible_polys.db")


class IrreduciblePolyDatabase:
    """
    Class to interface with the irreducible polynomials database.
    """

    def __new__(cls):
        global DATABASE
        if DATABASE is None:
            DATABASE = super().__new__(cls)
        return DATABASE

    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()

    def fetch(self, characteristic: int, degree: int):
        self.cursor.execute(
            """
            SELECT nonzero_degrees,nonzero_coeffs
            FROM polys
            WHERE characteristic=? AND degree=?""",
            (characteristic, degree),
        )
        result = self.cursor.fetchone()

        if result is None:
            raise LookupError(
                f"The irreducible polynomials database does not contain an entry for GF({characteristic}^{degree}).\n\n"
                "Alternatively, you can construct irreducible polynomials with `galois.irreducible_poly(p, m)` "
                "or primitive polynomials with `galois.primitive_poly(p, m)`."
            )

        nonzero_degrees = [int(_) for _ in result[0].split(",")]
        nonzero_coeffs = [int(_) for _ in result[1].split(",")]

        return nonzero_degrees, nonzero_coeffs
