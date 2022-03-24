import random

import numpy as np


def array_equal(a, b):
    # Weird NumPy comparison bug, see https://github.com/mhostetter/galois/issues/37
    if a.dtype == np.object_:
        return np.array_equal(a, np.array(b, dtype=np.object_))
    else:
        return np.array_equal(a, b)


def randint(low, high, shape, dtype):
    if np.issubdtype(dtype, np.integer):
        array = np.random.default_rng().integers(low, high, shape, dtype=np.int64)
    else:
        # For dtype=object
        array = np.empty(shape, dtype=dtype)
        iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
        for _ in iterator:
            array[iterator.multi_index] = random.randint(low, high - 1)
    return array
