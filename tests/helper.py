import random

import numpy as np

ALL_DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64, np.object_]


def randint(low, high, shape, dtype):
    if np.issubdtype(dtype, np.integer):
        array = np.random.randint(low, high, shape, dtype=np.int64)
    else:
        # For dtype=object
        array = np.empty(shape, dtype=dtype)
        iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
        for i in iterator:
            array[iterator.multi_index] = random.randint(low, high - 1)
    return array
