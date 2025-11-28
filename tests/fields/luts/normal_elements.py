"""
A module containing LUTs for normal elements.
"""

from .normal_elements_2 import (
    NORMAL_ELEMENTS_2_2,
    NORMAL_ELEMENTS_2_3,
    NORMAL_ELEMENTS_2_4,
    NORMAL_ELEMENTS_2_5,
    NORMAL_ELEMENTS_2_6,
)
from .normal_elements_3 import (
    NORMAL_ELEMENTS_3_2,
    NORMAL_ELEMENTS_3_3,
    NORMAL_ELEMENTS_3_4,
)
from .normal_elements_5 import (
    NORMAL_ELEMENTS_5_2,
    NORMAL_ELEMENTS_5_3,
    NORMAL_ELEMENTS_5_4,
)

NORMAL_ELEMENTS = [
    (2, 2, NORMAL_ELEMENTS_2_2),
    (2, 3, NORMAL_ELEMENTS_2_3),
    (2, 4, NORMAL_ELEMENTS_2_4),
    (2, 5, NORMAL_ELEMENTS_2_5),
    (2, 6, NORMAL_ELEMENTS_2_6),
    (3, 2, NORMAL_ELEMENTS_3_2),
    (3, 3, NORMAL_ELEMENTS_3_3),
    (3, 4, NORMAL_ELEMENTS_3_4),
    (5, 2, NORMAL_ELEMENTS_5_2),
    (5, 3, NORMAL_ELEMENTS_5_3),
    (5, 4, NORMAL_ELEMENTS_5_4),
]
