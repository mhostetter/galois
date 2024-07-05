"""
A module that handles interfacing with the SQLite databases.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from threading import Lock


class DatabaseInterface:
    """
    An abstract class to interface with SQLite databases.
    """

    _lock = Lock()
    _singleton = None
    file: Path

    def __new__(cls):
        with cls._lock:
            if cls._singleton is None:
                cls._singleton = super().__new__(cls)
                cls.conn = sqlite3.connect(cls.file, check_same_thread=False)
                cls.cursor = cls.conn.cursor()
        return cls._singleton


class PrimeFactorsDatabase(DatabaseInterface):
    """
    A class to interface with the prime factors database.
    """

    file = Path(__file__).parent / "prime_factors.db"

    def fetch(self, n: int) -> tuple[list[int], list[int], int]:
        """
        Fetches the prime factors and multiplicities of the given integer.

        Arguments:
            n: An integer.

        Returns:
            A tuple containing the prime factors and multiplicities.
        """
        # Integer string conversion length limitation since Python 3.11
        # https://github.com/mhostetter/galois/issues/494
        if hasattr(sys, "set_int_max_str_digits"):
            default_limit = sys.get_int_max_str_digits()
            sys.set_int_max_str_digits(0)
            n = str(n)
            sys.set_int_max_str_digits(default_limit)

        with self._lock:
            self.cursor.execute(
                """
                SELECT factors, multiplicities, composite
                FROM factorizations
                WHERE value=?
                """,
                (str(n),),
            )
            result = self.cursor.fetchone()

        if result is None:
            raise LookupError(f"The prime factors database does not contain an entry for {n}.")

        factors = [int(x) for x in result[0].split(",")]
        multiplicities = [int(x) for x in result[1].split(",")]
        composite = int(result[2])

        return factors, multiplicities, composite


class IrreduciblePolyDatabase(DatabaseInterface):
    """
    A class to interface with the irreducible polynomials database.
    """

    file = Path(__file__).parent / "irreducible_polys.db"

    def fetch(self, characteristic: int, degree: int) -> tuple[list[int], list[int]]:
        """
        Fetches the irreducible polynomial of degree `degree` over GF(`characteristic`).

        Arguments:
            characteristic: The prime characteristic of the field.
            degree: The degree of the polynomial.

        Returns:
            A tuple containing the non-zero degrees and coefficients of the irreducible polynomial.
        """
        with self._lock:
            self.cursor.execute(
                """
                SELECT nonzero_degrees, nonzero_coeffs
                FROM polys
                WHERE characteristic=? AND degree=?""",
                (characteristic, degree),
            )
            result = self.cursor.fetchone()

        if result is None:
            raise LookupError(
                f"The irreducible polynomials database does not contain an entry for a degree-{degree} polynomial "
                f"over GF({characteristic})."
            )

        nonzero_degrees = [int(_) for _ in result[0].split(",")]
        nonzero_coeffs = [int(_) for _ in result[1].split(",")]

        return nonzero_degrees, nonzero_coeffs


class ConwayPolyDatabase(DatabaseInterface):
    """
    A class to interface with the Conway polynomials database.
    """

    file = Path(__file__).parent / "conway_polys.db"

    def fetch(self, characteristic: int, degree: int) -> tuple[list[int], list[int]]:
        """
        Fetches the Conway polynomial of degree `degree` over GF(`characteristic`).

        Arguments:
            characteristic: The prime characteristic of the field.
            degree: The degree of the polynomial.

        Returns:
            A tuple containing the non-zero degrees and coefficients of the Conway polynomial.
        """
        with self._lock:
            self.cursor.execute(
                """
                SELECT nonzero_degrees, nonzero_coeffs
                FROM polys
                WHERE characteristic=? AND degree=?
                """,
                (characteristic, degree),
            )
            result = self.cursor.fetchone()

        if result is None:
            raise LookupError(
                f"Frank Luebeck's database of Conway polynomials does not contain an entry for a degree-{degree} "
                f"polynomial over GF({characteristic}). "
                f"See http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html for his complete list "
                "of polynomials.\n\n"
                "Alternatively, you can find irreducible polynomials with `galois.irreducible_poly(p, m)` "
                "or primitive polynomials with `galois.primitive_poly(p, m)`."
            )

        nonzero_degrees = [int(_) for _ in result[0].split(",")]
        nonzero_coeffs = [int(_) for _ in result[1].split(",")]

        return nonzero_degrees, nonzero_coeffs
