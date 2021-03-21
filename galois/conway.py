import os
import sqlite3

DATABASE = None  # Database singleton class
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "..", "databases", "conway_polys.db")


class ConwayDatabase:
    """
    Class to interface with the Conway polynomials database.
    """

    def __new__(cls):
        if DATABASE is None:
            return super().__new__(cls)
        else:
            return DATABASE

    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()

    def fetch(self, characteristic, degree):
        self.cursor.execute("SELECT coefficients FROM polys WHERE characteristic=? AND degree=?", (int(characteristic), int(degree)))
        result = self.cursor.fetchone()
        if result is not None:
            result = result[0]
        return result
