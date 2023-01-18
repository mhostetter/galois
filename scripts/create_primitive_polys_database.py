"""
A script to create a database of primitive polynomials

Sources:
 - Wolfram Research, "Primitive Polynomials" from the Wolfram Data Repository (2017): https://doi.org/10.24097/wolfram.48521.data

"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import requests
import hashlib
import json


def main():
    """
    The main routine to create a database of primitive polynomials
    """

    database_file = Path(__file__).parent.parent / "src" / "galois" / "_databases" / "primitive_polys.db"
    conn, cursor = create_database(database_file)

    _add_wolfram_2017(conn, cursor)

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
    cursor: sqlite3.Cursor, characteristic: int, degree: int, nonzero_degrees: str, nonzero_coeffs: str
):
    """
    Adds the given primitive polynomial to the database.
    """
    cursor.execute(
        """
        INSERT INTO polys (characteristic, degree, nonzero_degrees, nonzero_coeffs)
        VALUES (?,?,?,?)
        """,
        (characteristic, degree, nonzero_degrees, nonzero_coeffs),
    )


def _add_wolfram_2017(conn, cursor):
    """
    Add Wolfram's primitive polynomials to the database.
    Up to GF(2^1_200), GF(3^660), GF(5^430) and GF(7^358)
    """
    url = "https://www.wolframcloud.com/objects/8a6cda66-58d7-49cf-8a1b-5d4788ff6c6e"
    data = requests.get(url, stream=True).content
    sha256 = hashlib.sha256()
    sha256.update(data)
    assert sha256.hexdigest() == "38249bff65eb06d74b9188ccbf4c29fe2becd0716588bf78d418b31463d30703"

    data = json.loads(data)

    print("Parsing Wolfram's primitive polynomials (2017)...")

    for entry in data[1:]:
        characteristic = entry[1][1]
        degree = entry[1][2]
        coeffs = entry[2][1][2]
        nonzero_degrees = [0]
        nonzero_coeffs = [coeffs[1]]
        for term in coeffs[2:]:
            if term[0] == "Power":
                nonzero_degrees += [term[2]]
                nonzero_coeffs += [1]
            elif term[0] == "Times":
                if term[2][0] == "Power":
                    nonzero_degrees += [term[2][2]]
                    nonzero_coeffs += [term[1]]
                else:  # Case P(x) = n * x
                    nonzero_degrees += [1]
                    nonzero_coeffs += [term[1]]
            else:  # Case P(x) = x
                nonzero_degrees += [1]
                nonzero_coeffs += [1]
        nonzero_degrees = str(nonzero_degrees[::-1]).replace(" ", "")[1:-1]
        nonzero_coeffs = str(nonzero_coeffs[::-1]).replace(" ", "")[1:-1]
        print(f"Irreducible polynomial for GF({characteristic}^{degree})")
        add_to_database(cursor, characteristic, degree, nonzero_degrees, nonzero_coeffs)

    conn.commit()


if __name__ == "__main__":
    main()
