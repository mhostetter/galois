import random

import numpy as np


DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64, np.object_]


def valid_dtype(field):
    return random.choice(field.dtypes)


def invalid_dtype(field):
    return random.choice([dtype for dtype in DTYPES if dtype not in field.dtypes])
