"""
A script to create a database of irreducible polynomials

Sources:
 - Gadiel Seroussi. Table of Low-Weight Binary Irreducible Polynomials (1998): https://www.hpl.hp.com/techreports/98/HPL-98-135.html

"""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from pathlib import Path

import requests
from pdfminer.high_level import extract_text


def main():
    """
    The main routine to create a database of irreducible polynomials
    """

    database_file = Path(__file__).parent.parent / "src" / "galois" / "_databases" / "irreducible_polys.db"
    conn, cursor = create_database(database_file)

    _add_hpl_1998(conn, cursor)

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
    Adds the given irreducible polynomial to the database.
    """
    cursor.execute(
        """
        INSERT INTO polys (characteristic, degree, nonzero_degrees, nonzero_coeffs)
        VALUES (?,?,?,?)
        """,
        (characteristic, degree, nonzero_degrees, nonzero_coeffs),
    )


def _add_hpl_1998(conn, cursor):
    """
    Add Gadiel Seroussi's table to the database.
    GF(2^m) for 2 <= m <= 10_000
    """
    url = "https://www.hpl.hp.com/techreports/98/HPL-98-135.pdf"
    # There is an issue with the SSL certificate using CURL_CA_BUNDLE
    # We don't validate https, but we do check the PDF's checksum
    pdf = requests.get(url, stream=True, verify=False).content
    sha256 = hashlib.sha256()
    sha256.update(pdf)
    assert sha256.hexdigest() == "78f02d84a0957ad261c53a0d1107adb2ff9d72f52ba5e10ea77eaa8cf766a0ee"

    coefficients = []
    print("Parsing Table of Low-Weight Binary Irreducible Polynomials (1998)...")
    for page in range(3, 16):
        text = extract_text(io.BytesIO(pdf), page_numbers=[page])  # extract_text doesn't accept Bytes as input
        # Tabs are parsed as \n\n, except when the irreducible poly is a pentanomial.
        # In that case, there is only a space. First replace takes care of that.
        # Second replace unifies tabs and changes of lines.
        # Every page ends with the page number and the form feed \x0c, hence the [:-2].
        coefficients += text.replace(" ", "\n").replace("\n\n", "\n").split("\n")[:-2]

    for coeffs in coefficients:
        degree = coeffs.split(",")[0]
        nonzero_degrees = coeffs + ",0"
        nonzero_coeffs = ("1," * len(nonzero_degrees.split(",")))[:-1]
        print(f"Irreducible polynomial for GF(2^{degree})")
        add_to_database(cursor, 2, degree, nonzero_degrees, nonzero_coeffs)

    conn.commit()


if __name__ == "__main__":
    main()
