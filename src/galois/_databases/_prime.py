"""
A module that handles accessing the database of prime factors.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DATABASE = None  # Database singleton class
DATABASE_FILE = Path(__file__).parent / "prime_factors.db"


class PrimeFactorsDatabase:
    """
    Class to interface with the prime factors database.
    """

    def __new__(cls):
        global DATABASE
        if DATABASE is None:
            DATABASE = super().__new__(cls)
        return DATABASE

    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()

    def fetch(self, n: int) -> tuple[list[int], list[int], int]:
        self.cursor.execute("SELECT factors, multiplicities, composite FROM factorizations WHERE value=?", (str(n),))
        result = self.cursor.fetchone()

        if result is None:
            raise LookupError

        factors = [int(x) for x in result[0].split(",")]
        multiplicities = [int(x) for x in result[1].split(",")]
        composite = int(result[2])

        return factors, multiplicities, composite
