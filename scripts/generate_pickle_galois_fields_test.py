import pickle
from pathlib import Path
from typing import Any

import galois

FOLDER = Path(__file__).parent.parent / "tests" / "data"


def save_pickle(data: Any, name: str) -> None:
    print(f"  Saving {FOLDER / name}...")
    with open(FOLDER / name, "wb") as f:
        pickle.dump(data, f)


def generate_test_pickling_of_galois_fields():
    """
    Generates pickle data for testing pickling of Galois fields.

    The data generated here is used in the test_pickling_of_fields() test function.
    We generate various fields with various primitive elements and polynomials. When
    we read them back in, we verify that the serialization is correct.
    """

    # Prime from bug report #539.  BLS12-381 characteristic
    bug539_prime = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001
    data = []
    # Look at some prime fields, including the prime from the bug report
    for prime in (2, 5, 7, 31, 3191, 2**31 - 1, bug539_prime):
        if prime == bug539_prime:
            # Generating a primitive root for this large value can take several minutes.
            # Verifying the field is also slow.
            alpha, verify = 1002, False
        else:
            alpha, verify = galois.primitive_root(prime, method="random"), True
        field = galois.GF(prime, 1, primitive_element=alpha, verify=verify)
        info = dict(field=field, order=prime, primitive_element=int(alpha))
        data.append(info)

    # Look at various prime powers
    for prime, degree in ((2, 2), (2, 3), (2, 8), (2, 32), (2, 100), (7, 3), (109987, 4)):
        poly = galois.irreducible_poly(prime, degree, method="random")
        alpha = galois.primitive_element(poly, method="random")
        field = galois.GF(prime, degree, primitive_element=alpha, irreducible_poly=poly)
        info = dict(field=field, order=(prime**degree), primitive_element=int(alpha), irreducible_poly=str(poly))
        data.append(info)

    # Save all of this info to a pickle file.
    save_pickle(data, "field_pickle.pkl")


if __name__ == "__main__":
    generate_test_pickling_of_galois_fields()
