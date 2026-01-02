"""
A pytest module to test pickling and unpickling of FieldArray subclasses and their instances.
"""

import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

import galois


@pytest.mark.parametrize("order", [3, 3**2])
def test_pickle_field_array_in_same_interpreter(order, tmp_path):
    GF = galois.GF(order)
    x = GF.Random(10)

    # Write the pickle artifact
    pkl_path = tmp_path / "field_array.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(x, f)

    # Read the pickle artifact
    with pkl_path.open("rb") as f:
        x2 = pickle.load(f)

    assert type(x) is type(x2)
    assert np.array_equal(x, x2)


@pytest.mark.parametrize("order", [3, 3**2])
def test_pickle_field_array_in_new_interpreter(order, tmp_path):
    GF = galois.GF(order)
    x = GF.Random(10)

    # Write the pickle artifact
    pkl_path = tmp_path / "field_array.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(x, f)

    # Capture "expected" in a representation that is easy to embed as Python literals.
    # Use integers of the underlying representation, not repr(x).
    expected_hash = hash(type(x))
    expected_properties = type(x).properties
    expected_values = x.tolist()

    # Run a fresh interpreter that ONLY imports galois and unpickles.
    code = f"""
import pickle
import numpy as np
import galois

with open(r"{pkl_path}", "rb") as f:
    x2 = pickle.load(f)

assert hash(type(x2)) == {expected_hash}
assert type(x2).properties == {expected_properties!r}
assert x2.tolist() == {expected_values!r}
"""

    env = os.environ.copy()

    # Ensure subprocess imports your working tree version (editable checkouts, etc.)
    repo_root = os.getcwd()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Unpickling in a fresh interpreter failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
    )
