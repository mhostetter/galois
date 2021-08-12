"""
A module that contains the list of valid integer dtypes for Galois field arrays. This global is imported in many other modules.
"""
import numpy as np

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]
