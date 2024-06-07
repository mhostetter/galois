"""
A script to create a database of Conway polynomials using Frank Luebeck's compilation of polynomials.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np
import requests


def main():
    """
    The main routine to create a database of Conway polynomials.
    """
    content = requests.get("http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CPimport.txt").content
    text = content.decode("utf-8")

    # Verify the document is the same as expected
    sha256 = hashlib.sha256()
    sha256.update(content)
    assert sha256.hexdigest() == "012114e531cdde43b2b1f064c3f2f548ecc755e08b82971ce9ccbeeb652dc159"

    database_file = Path(__file__).parent.parent / "src" / "galois" / "_databases" / "conway_polys.db"
    conn, cursor = create_database(database_file)

    for line in text.splitlines():
        # Skip first and last lines
        if line in ["allConwayPolynomials := [", "0];"]:
            continue

        line = line[1:-2]  # Trim leading and trailing brackets
        items = line.split(",", maxsplit=2)
        characteristic = int(items[0])
        degree = int(items[1])

        # Degree-descending coefficients
        coeffs = np.array(eval(items[2])[::-1])

        idxs = np.nonzero(coeffs)[0]
        nonzero_degrees = (degree - idxs).tolist()
        nonzero_coeffs = coeffs[idxs].tolist()

        print(f"Degree-{degree} Conway polynomial over GF({characteristic})")
        print(f"  {nonzero_degrees}, {nonzero_coeffs}")

        add_to_database(cursor, characteristic, degree, nonzero_degrees, nonzero_coeffs)

    conn.commit()
    conn.close()


def create_database(file: Path) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Deletes the old database, makes a new one, and returns the database connection.
    """
    if file.exists():
        os.remove(file)

    conn = sqlite3.connect(file)
    cursor = conn.cursor()
    create_table(conn, cursor)

    return conn, cursor


def create_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Creates an empty 'polys' table.
    """
    cursor.execute(
        """
        CREATE TABLE polys (
            characteristic INTEGER NOT NULL,
            degree INTEGER NOT NULL,
            nonzero_degrees TEXT NOT NULL,
            nonzero_coeffs TEXT NOT NULL,
            PRIMARY KEY (characteristic, degree)
        )
        """
    )
    conn.commit()


def add_to_database(
    cursor: sqlite3.Cursor,
    characteristic: int,
    degree: int,
    nonzero_degrees: list[int],
    nonzero_coeffs: list[int],
):
    """
    Adds the given Conway polynomial to the database.
    """
    degrees = ",".join(str(d) for d in nonzero_degrees)
    coeffs = ",".join(str(c) for c in nonzero_coeffs)

    cursor.execute(
        """
        INSERT INTO polys (characteristic, degree, nonzero_degrees, nonzero_coeffs)
        VALUES (?,?,?,?)
        """,
        (characteristic, degree, degrees, coeffs),
    )


if __name__ == "__main__":
    main()
