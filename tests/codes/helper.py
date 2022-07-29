import random

import numpy as np


def random_errors(GF, N, n, max_errors):
    max_errors = min(n, max_errors)
    N_errors = np.random.default_rng().integers(0, max_errors + 1, N)
    N_errors[0] = max_errors  # Ensure the max number of errors is present at least once

    E = GF.Zeros((N, n))
    for i in range(N):
        E[i, random.sample(list(range(n)), N_errors[i])] = GF.Random(N_errors[i], low=1)

    return E, N_errors


def random_type(array):
    """
    Randomly vary the input type to encode()/decode() across various ArrayLike inputs.
    """
    x = random.randint(0, 2)
    if x == 0:
        # A FieldArray instance
        return array
    elif x == 1:
        # A np.ndarray instance
        return array.view(np.ndarray)
    else:
        return array.tolist()
