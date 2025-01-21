"""
A script to create a database of prime factorizations of p^n +/- 1 using the Cunningham Project's tables.
https://homes.cerias.purdue.edu/~ssw/cun/
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import requests

import galois


def main():
    """
    The main routine to create a database of prime factorizations of b^n +/- 1.
    """
    primes = create_prime_lut()
    composites = create_composite_lut()

    database_file = Path(__file__).parent.parent / "src" / "galois" / "_databases" / "prime_factors.db"
    conn, cursor = create_database(database_file)

    add_main_tables(conn, cursor, primes, composites)

    for base in [2, 3, 5, 6, 7, 10, 11, 12]:
        create_even_negative_offset_table(conn, cursor, base)

    conn.close()


def create_prime_lut() -> dict[tuple[str, str], int]:
    """
    Creates a dictionary of special prime numbers. The key is (label, name). For example, ("2,471+", "P21").
    """
    primes = process_appendix_a()
    extra_primes = process_other_primes()

    # There are duplicate entries in these tables and they contain different primes! So we're saving both
    # and figuring out which one is correct at runtime.
    for key, value in extra_primes.items():
        if key in primes:
            primes[key] = [primes[key], value]
        else:
            primes[key] = value

    return primes


def process_appendix_a() -> dict[tuple[str, str], int]:
    """
    Process Appendix A that contains large primes, e.g. P52.
    """
    url = "https://homes.cerias.purdue.edu/~ssw/cun/third/appa901"
    text = requests.get(url).text

    # Make multi-line entries single-line
    text = re.sub("\n                   ", "", text)

    primes = {}
    for line in text.splitlines()[1:]:
        if line.startswith("     There"):
            continue
        bits = int(line[0:4].strip())
        label = line[4:14].strip()
        assert line[14:19] == "P    "
        prime_str = line[19:].replace(" ", "")
        assert len(prime_str) == bits
        primes[(label, f"P{bits}")] = int(prime_str)

    return primes


def process_other_primes() -> dict[tuple[str, str], int]:
    """
    Process all primes in the main tables (except those already in Appendix A).
    """
    text = requests.get("https://homes.cerias.purdue.edu/~ssw/cun/third/mainprimes").text

    primes = {}
    for line in text.splitlines()[1:]:
        bits = int(line[0:4].strip())
        label = line[4:18].strip()
        prime_str = line[18:].replace(" ", "")
        assert len(prime_str) == bits
        primes[(label, f"P{bits}")] = int(prime_str)

    return primes


def create_composite_lut() -> dict[tuple[str, str], int]:
    """
    Creates a dictionary of special composite numbers. The key is (label, name). For example, ("2,471+", "C152").
    """
    return process_appendix_c()


def process_appendix_c() -> dict[tuple[str, str], int]:
    """
    Process Appendix C that contains large composites, e.g. C152.
    """
    text = requests.get("https://homes.cerias.purdue.edu/~ssw/cun/third/appc901").text

    # Make multi-line entries single-line
    text = re.sub("\n                   ", "", text)

    last_key = None

    composites = {}
    for line in text.splitlines()[1:]:
        bits = int(line[0:4].strip())
        label = line[4:16].replace(" ", "")
        partial_composite_str = line[16:].replace(" ", "")

        if label != "":
            key = (label, f"C{bits}")
            composites[key] = int(partial_composite_str)
            last_key = key
        else:
            assert last_key is not None
            assert last_key in composites
            composites[last_key] = int(str(composites[last_key]) + partial_composite_str)

    # Test that the stitching worked
    for key, value in composites.items():
        bits = int(key[1][1:])
        assert len(str(value)) == bits

    return composites


def create_database(file: Path) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Deletes the old database, makes a new one, and returns the database connection.
    """
    if file.exists():
        os.remove(file)

    conn = sqlite3.connect(file)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    create_table(conn, cursor)

    return conn, cursor


def create_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Creates an empty 'factorizations' table.
    """
    cursor.execute(
        """
        CREATE TABLE factorizations (
            base INTEGER NOT NULL,
            exponent INTEGER NOT NULL,
            offset INTEGER NOT NULL,
            value TEXT NOT NULL,
            factors TEXT NOT NULL,
            multiplicities TEXT NOT NULL,
            composite TEXT NOT NULL,
            PRIMARY KEY (value)
        )
        """
    )
    conn.commit()


def add_main_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor, primes: dict, composites: dict):
    """
    Adds the main factorization tables of the Cunningham Book to the database.
    """
    text = requests.get("https://homes.cerias.purdue.edu/~ssw/cun/third/pmain901").text

    for lines in text.split("    Table ")[1:]:
        header, data = lines.split("n                                Prime Factors", 2)
        data = clean_up_data(data)

        pattern = r"(?P<name>[a-zA-Z0-9()+-]*)\s+Factorizations of (?P<base>\d+)\^n(?P<offset>[+-]\d+), (?P<extra>.*)"
        match = re.search(pattern, header)
        assert match is not None, f"Could not parse header: {header}"

        table = {
            "name": match.group("name"),
            "base": int(match.group("base")),
            "offset": int(match.group("offset")),
            "extra": match.group("extra"),
            "primes": primes,
            "composites": composites,
        }

        rows = {}
        for line in data.splitlines():
            if line in ["", " -------------------------------------------------------------------------------"]:
                continue
            if "^" in line:
                # These lines contain the relationship equations at the end of a table
                continue

            rows, row = add_to_rows(table, rows, line)

            base = table["base"]
            offset = table["offset"]

            exponent = row["n"]
            value = row["value"]
            factors = list(row["factors"].keys())
            multiplicities = list(row["factors"].values())
            composite = row["composite"]

            add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)

    conn.commit()


def clean_up_data(text: str) -> str:
    """
    Preprocess the text file to make it easier to further parse.
    """
    # Eliminate new lines that start with '.' since they indicate a continuation of the previous line
    text = re.sub(r"\.\n\s+", "", text)

    # Eliminate new lines that start with \ since they break up a long integer
    text = re.sub(r"\\\s+", "", text)

    # Eliminate spaces between '.' and the next prime
    text = re.sub(r"\.\s+", ".", text)

    # Eliminate new lines that also indent the next line since they indicate a continuation of the previous line
    text = re.sub(r"\n      ", "", text)

    # Convert new lines with L and M onto the same line. We will then use the '\tL' and '\tM' to split off those sections.
    text = re.sub(r"\n     L", "\tL", text)
    text = re.sub(r"\n     M", "\tM", text)

    return text


def add_to_rows(table: dict, rows: dict, line: str) -> tuple[dict, dict]:
    # Strip off the exponent 'n' first. It is always right-aligned and 5 characters long.
    n = int(line[0:5].strip())
    line = line[5:]

    if line[0] == "L":
        # This only happens in the 2LM table. Make the line look like: L.M \tL<l_section>\tM<m_section>
        line = "L.M \t" + line

    if "L.M" in line:
        assert "\tL" in line and "\tM" in line, "Line should have both L and M sections"
        line, l_section = line.split("\tL", 1)
        l_section, m_section = l_section.split("\tM", 1)

        # Process the L and M sections first
        rows, _ = add_section_data(table, rows, n, "L", l_section.strip())
        rows, _ = add_section_data(table, rows, n, "M", m_section.strip())

    rows, row = add_section_data(table, rows, n, "", line.strip())

    return rows, row


def add_section_data(table: dict, rows: dict, n: int, letter: str, section: str) -> tuple[dict, dict]:
    """
    Processes a factorization section, e.g. "(3L,21M,57M) 7928131.1262555546640315313.P92". The contents in the
    () indicate other factored values from the table that divide this value. After the () are the new factors
    for this value.
    """
    if letter in ["L", "M"]:
        key = f"{n}{letter}"
    else:
        key = n
    value = table["base"] ** n + table["offset"]

    row = {
        "value": value,
        "n": n,
        "factors": {},  # Key is the factor, value is the multiplicity
        "composite": value,  # The remaining composite number after factoring
    }

    divisors_str, factors_str = parse_section(section)
    row = parse_divisors_string(rows, row, divisors_str)
    row = parse_factors_string(table, rows, row, factors_str, letter)

    # Assign this row of the LUT to the table dictionary
    rows[key] = row

    return rows, row


def parse_section(section: str) -> tuple[str, str]:
    """
    Parses the divisors and factors sections. The divisors section are in () and the factors section is after.
    """
    if "(" in section:
        assert section.count("(") == 1 and section.count(")") == 1, "There should be exactly one pair of parentheses"

        _, divisors_str = section.split("(", 1)
        divisors_str, factors_str = divisors_str.split(")", 1)
        factors_str = factors_str.strip()
        return divisors_str, factors_str

    return "", section.strip()


def parse_divisors_string(rows: dict, row: dict, string: str) -> dict:
    """
    Processes the other values that divide this value. It checks each factor of the divisor value and adds it
    with its multiplicity to the current row.
    """
    for divisor in string.split(","):
        if divisor == "":
            continue

        try:
            key = int(divisor)
        except ValueError:
            key = divisor

        for factor in rows[key]["factors"]:
            row = add_factor(row, factor)

    return row


def parse_factors_string(table: dict, rows: dict, row: dict, string: str, letter: str) -> dict:
    """
    Processes the new factor of this value and adds it with its multiplicity to the current row.
    """
    # Remove extra spaces to aid in parsing
    string = re.sub(r" ", "", string)

    # Remove the '*' since those indicate repeated factors
    string = re.sub(r"\*", "", string)

    # Remove extra '.' that separate factors
    string = re.sub(r"\.+", ".", string)

    for factor in string.split("."):
        if "L" in factor or "M" in factor:
            # Must be an L or M sub-factor from the current table
            if factor in ["L", "M"]:
                key = f"{row['n']}{factor}"
            else:
                key = factor
            for factor in rows[key]["factors"]:
                row = add_factor(row, factor)
        elif "P" in factor:
            # A special prime
            label = prime_composite_label(table, row["n"], letter)
            key = (label, factor)
            factor = table["primes"][key]
            if isinstance(factor, int):
                row = add_factor(row, factor)
            else:
                # There are multiple entries for this key. We need to try each one.
                for f in factor:
                    if row["composite"] % f == 0:
                        row = add_factor(row, f)
        elif "C" in factor:
            # A special composite. Verify that it divides the remaining value.
            label = prime_composite_label(table, row["n"], letter)
            key = (label, factor)
            assert row["composite"] % table["composites"][key] == 0, (
                f"{row['composite']} is not divisible by {table['composites'][key]}"
            )
        else:
            # Must be a regular integer
            factor = int(factor)
            row = add_factor(row, factor)

    return row


def prime_composite_label(table: dict, n: int, letter: str) -> str:
    """
    Given the current table, the exponent 'n', and the letter 'L' or 'M', this function determines the label given
    to the special prime or composite.
    """
    if letter in ["L", "M"]:
        return f"{table['base']},{n}{letter}"
    if table["offset"] == 1:
        return f"{table['base']},{n}+"
    if table["offset"] == -1:
        return f"{table['base']},{n}-"

    raise RuntimeError(f"Could not get label for table: {table}")


def add_factor(row: dict, factor: int) -> dict:
    """
    Adds the given prime factor to the row and finds its multiplicity. This function reduces the remaining
    value 'composite' accordingly.
    """
    while row["composite"] % factor == 0:
        row["composite"] //= factor
        if factor not in row["factors"]:
            row["factors"][factor] = 0
        row["factors"][factor] += 1

    return row


def add_to_database(
    cursor: sqlite3.Cursor,
    base: int,
    exponent: int,
    offset: int,
    value: int,
    factors: list[int],
    multiplicities: list[int],
    composite: int,
):
    """
    Add the given factorization to the database. This function verifies the validity of the factorization before
    writing to the database.
    """
    if len(factors) > 0:
        # Sort the factors and multiplicities by ascending factor
        factors, multiplicities = zip(*sorted(zip(factors, multiplicities), key=lambda pair: pair[0]))

    test_factorization(base, exponent, offset, value, factors, multiplicities, composite)

    factors_str = ",".join([str(f) for f in factors])
    multiplicities_str = ",".join([str(m) for m in multiplicities])

    print(f"Adding to database: {base}^{exponent} + {offset}")

    # print("----------------------------------------------------------------------------------")
    # print("Adding to database:")
    # print(f"  value: {base}^{exponent} + {offset} = {value}")
    # print(f"  factors: {factors_str}")
    # print(f"  multiplicities: {multiplicities_str}")
    # print(f"  composite: {composite}")

    # NOTE: Ignore duplicates, which rarely occur. For example, 3^1 + 1 = 4 = 5^1 - 1.
    cursor.execute(
        """
        INSERT OR IGNORE INTO factorizations (base, exponent, offset, value, factors, multiplicities, composite)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (base, exponent, offset, str(value), factors_str, multiplicities_str, str(composite)),
    )


def test_factorization(
    base: int, exponent: int, offset: int, value: int, factors: list[int], multiplicities: list[int], composite: int
):
    """
    Tests that all the factorization parameters are consistent.
    """
    assert base**exponent + offset == value, f"{base}^{exponent} + {offset} != {value}"
    assert not galois.is_prime(composite), f"{composite} is prime"

    product = composite
    for factor, multiplicity in zip(factors, multiplicities):
        product *= factor**multiplicity
    assert product == value, f"{product} != {value}"


def create_even_negative_offset_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor, base: int):
    """
    Creates database entries for base^(2k) - 1 = A * B, A = (base^k - 1), B = (base^k + 1).
    """
    if base == 2:
        seed_two_even_negative_offset_tables(conn, cursor)
        k_fail = 1201
    elif base == 3:
        seed_three_even_negative_offset_tables(conn, cursor)
        k_fail = 541
    elif base == 5:
        seed_five_even_negative_offset_tables(conn, cursor)
        k_fail = 376
    elif base == 6:
        seed_six_even_negative_offset_tables(conn, cursor)
        k_fail = 331
    elif base == 7:
        seed_seven_even_negative_offset_tables(conn, cursor)
        k_fail = 301
    elif base == 10:
        seed_ten_even_negative_offset_tables(conn, cursor)
        k_fail = 331
    elif base == 11:
        seed_eleven_even_negative_offset_tables(conn, cursor)
        k_fail = 241
    elif base == 12:
        seed_twelve_even_negative_offset_tables(conn, cursor)
        k_fail = 241
    else:
        raise ValueError(f"Invalid base: {base}")

    k = 2
    while True:
        A_value = base**k - 1
        B_value = base**k + 1

        row = select_two_factorizations_from_database(cursor, A_value, B_value)
        if row is None:
            assert k == k_fail, (
                f"The {base}^(2k) - 1 table generation failed at k = {k}, but should have failed at k = {k_fail}"
            )
            break

        exponent = 2 * k
        offset = -1
        value = base**exponent + offset
        factors, multiplicities = merge_factors(row)
        composite = int(row["A_composite"]) * int(row["B_composite"])

        add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
        conn.commit()  # Need to commit after each row because it is used in the next iteration

        k += 1


def seed_two_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 2^2 - 1 factorization to the database.
    """
    base = 2
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [3]
    multiplicities = [1]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def seed_three_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 3^2 - 1 factorization to the database.
    """
    base = 3
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [2]
    multiplicities = [3]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def seed_five_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 5^2 - 1 factorization to the database.
    """
    base = 5
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [2, 3]
    multiplicities = [3, 1]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def seed_six_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 6^2 - 1 factorization to the database.
    """
    base = 6
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [5, 7]
    multiplicities = [1, 1]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def seed_seven_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 7^2 - 1 factorization to the database.
    """
    base = 7
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [2, 3]
    multiplicities = [4, 1]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def seed_ten_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 10^2 - 1 factorization to the database.
    """
    base = 10
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [3, 11]
    multiplicities = [2, 1]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def seed_eleven_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 11^2 - 1 factorization to the database.
    """
    base = 11
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [2, 3, 5]
    multiplicities = [3, 1, 1]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def seed_twelve_even_negative_offset_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """
    Manually adds the 12^2 - 1 factorization to the database.
    """
    base = 12
    exponent = 2
    offset = -1
    value = base**exponent + offset
    factors = [11, 13]
    multiplicities = [1, 1]
    composite = 1
    add_to_database(cursor, base, exponent, offset, value, factors, multiplicities, composite)
    conn.commit()


def select_two_factorizations_from_database(cursor: sqlite3.Cursor, A_value: int, B_value: int):
    """
    Selects two factorizations from the database.
    """
    cursor.execute(
        """
        SELECT
            A.factors AS A_factors,
            A.multiplicities AS A_multiplicities,
            A.composite AS A_composite,
            B.factors AS B_factors,
            B.multiplicities AS B_multiplicities,
            B.composite AS B_composite
        FROM
            factorizations A,
            factorizations B
        WHERE (
            A.value == ? AND B.value == ?
        )
        """,
        (str(A_value), str(B_value)),
    )

    return cursor.fetchone()


def merge_factors(row: sqlite3.Row) -> tuple[list[int], list[int]]:
    """
    Combines the factors and multiplicities from two factorizations.
    """
    factors = []
    multiplicities = []
    if row["A_factors"] != "":
        factors += [int(f) for f in row["A_factors"].split(",")]
    if row["A_multiplicities"] != "":
        multiplicities = [int(m) for m in row["A_multiplicities"].split(",")]
    if row["B_factors"] != "":
        factors += [int(f) for f in row["B_factors"].split(",")]
    if row["B_multiplicities"] != "":
        multiplicities += [int(m) for m in row["B_multiplicities"].split(",")]
    return factors, multiplicities


if __name__ == "__main__":
    main()
