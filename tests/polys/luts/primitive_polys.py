"""
A module containing LUTs for primitive polynomials.
"""

from .primitive_polys_2 import (
    PRIMITIVE_POLYS_2_1,
    PRIMITIVE_POLYS_2_2,
    PRIMITIVE_POLYS_2_3,
    PRIMITIVE_POLYS_2_4,
    PRIMITIVE_POLYS_2_5,
    PRIMITIVE_POLYS_2_6,
    PRIMITIVE_POLYS_2_7,
    PRIMITIVE_POLYS_2_8,
)
from .primitive_polys_3 import (
    PRIMITIVE_POLYS_3_1,
    PRIMITIVE_POLYS_3_2,
    PRIMITIVE_POLYS_3_3,
    PRIMITIVE_POLYS_3_4,
    PRIMITIVE_POLYS_3_5,
    PRIMITIVE_POLYS_3_6,
)
from .primitive_polys_4 import (
    PRIMITIVE_POLYS_4_1,
    PRIMITIVE_POLYS_4_2,
    PRIMITIVE_POLYS_4_3,
)
from .primitive_polys_5 import (
    PRIMITIVE_POLYS_5_1,
    PRIMITIVE_POLYS_5_2,
    PRIMITIVE_POLYS_5_3,
    PRIMITIVE_POLYS_5_4,
)
from .primitive_polys_9 import (
    PRIMITIVE_POLYS_9_1,
    PRIMITIVE_POLYS_9_2,
    PRIMITIVE_POLYS_9_3,
)
from .primitive_polys_25 import PRIMITIVE_POLYS_25_1, PRIMITIVE_POLYS_25_2

PRIMITIVE_POLYS = [
    (2, 1, PRIMITIVE_POLYS_2_1),
    (2, 2, PRIMITIVE_POLYS_2_2),
    (2, 3, PRIMITIVE_POLYS_2_3),
    (2, 4, PRIMITIVE_POLYS_2_4),
    (2, 5, PRIMITIVE_POLYS_2_5),
    (2, 6, PRIMITIVE_POLYS_2_6),
    (2, 7, PRIMITIVE_POLYS_2_7),
    (2, 8, PRIMITIVE_POLYS_2_8),
    (2**2, 1, PRIMITIVE_POLYS_4_1),
    (2**2, 2, PRIMITIVE_POLYS_4_2),
    (2**2, 3, PRIMITIVE_POLYS_4_3),
    (3, 1, PRIMITIVE_POLYS_3_1),
    (3, 2, PRIMITIVE_POLYS_3_2),
    (3, 3, PRIMITIVE_POLYS_3_3),
    (3, 4, PRIMITIVE_POLYS_3_4),
    (3, 5, PRIMITIVE_POLYS_3_5),
    (3, 6, PRIMITIVE_POLYS_3_6),
    (3**2, 1, PRIMITIVE_POLYS_9_1),
    (3**2, 2, PRIMITIVE_POLYS_9_2),
    (3**2, 3, PRIMITIVE_POLYS_9_3),
    (5, 1, PRIMITIVE_POLYS_5_1),
    (5, 2, PRIMITIVE_POLYS_5_2),
    (5, 3, PRIMITIVE_POLYS_5_3),
    (5, 4, PRIMITIVE_POLYS_5_4),
    (5**2, 1, PRIMITIVE_POLYS_25_1),
    (5**2, 2, PRIMITIVE_POLYS_25_2),
]
