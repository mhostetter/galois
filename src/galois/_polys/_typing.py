"""
A module containing type hints for the polynomial-related objects.
"""
from __future__ import annotations

from typing import Union, TYPE_CHECKING

from .._domains._typing import ArrayLike

# Obtain forward references
if TYPE_CHECKING:
    from .._polys._poly import Poly

__all__ = ["PolyLike"]


PolyLike = Union[int, str, ArrayLike, "Poly"]
"""
A :obj:`~typing.Union` representing objects that can be coerced into a polynomial.

:group: polys

.. rubric:: Union

- :obj:`int`: A polynomial in its integer representation, see :func:`~galois.Poly.Int`. The Galois field must be known from context.

  .. ipython:: python

      # Known from context
      GF = galois.GF(3)
      galois.Poly.Int(19, field=GF)

- :obj:`str`: A polynomial in its string representation, see :func:`~galois.Poly.Str`. The Galois field must be known from context.

  .. ipython:: python

      galois.Poly.Str("2x^2 + 1", field=GF)

- :obj:`~galois.typing.ArrayLike`: An array of polynomial coefficients in degree-descending order. If the coefficients are not
  :obj:`~galois.Array`, then the Galois field must be known from context.

  .. ipython:: python

      galois.Poly([2, 0, 1], field=GF)
      galois.Poly(GF([2, 0, 1]))

- :obj:`~galois.Poly`: A previously-created :obj:`~galois.Poly` object. No coercion is necessary.

.. rubric:: Alias
"""
