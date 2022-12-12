"""
Script to create a database of Conway polynomials using Frank Luebeck's compilation of polynomials.
"""
import os
import sqlite3
from pathlib import Path

import requests

POLY_TEXT_FILE_URL = "http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CPimport.txt"
DATABASE_FILE = Path(__file__).parent.parent / "src" / "galois" / "_databases" / "conway_polys.db"

if DATABASE_FILE.exists():
    os.remove(DATABASE_FILE)

conn = sqlite3.connect(DATABASE_FILE)
cursor = conn.cursor()

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

text = requests.get(POLY_TEXT_FILE_URL).text
for line in text.splitlines():
    # Skip first and last lines
    if line in ["allConwayPolynomials := [", "0];"]:
        continue

    line = line[1:-2]  # Trim leading and trailing brackets
    characteristic, degree, coefficients = line.split(",", maxsplit=2)
    print(f"Conway polynomial for GF({characteristic}^{degree})")

    cursor.execute(
        """
        INSERT INTO polys (characteristic, degree, coefficients)
        VALUES (?,?,?)
        """,
        (int(characteristic), int(degree), coefficients),
    )

conn.commit()
conn.close()
