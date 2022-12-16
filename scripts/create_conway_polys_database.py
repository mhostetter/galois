"""
A script to create a database of Conway polynomials using Frank Luebeck's compilation of polynomials.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import requests


def main():
    """
    The main routine to create a database of Conway polynomials.
    """
    text = requests.get("http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CPimport.txt").text

    database_file = Path(__file__).parent.parent / "src" / "galois" / "_databases" / "conway_polys.db"
    conn, cursor = create_database(database_file)

    for line in text.splitlines():
        # Skip first and last lines
        if line in ["allConwayPolynomials := [", "0];"]:
            continue

        line = line[1:-2]  # Trim leading and trailing brackets
        characteristic, degree, coefficients = line.split(",", maxsplit=2)
        print(f"Conway polynomial for GF({characteristic}^{degree})")

        add_to_database(cursor, characteristic, degree, coefficients)

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
            coefficients TEXT NOT NULL,
            PRIMARY KEY (characteristic, degree)
        )
        """
    )
    conn.commit()


def add_to_database(cursor: sqlite3.Cursor, characteristic: str, degree: str, coefficients: str):
    """
    Adds the given Conway polynomial to the database.
    """
    cursor.execute(
        """
        INSERT INTO polys (characteristic, degree, coefficients)
        VALUES (?,?,?)
        """,
        (int(characteristic), int(degree), coefficients),
    )


if __name__ == "__main__":
    main()
