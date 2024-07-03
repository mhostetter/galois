"""
A public module containing type hints for the :obj:`galois` library.
"""
# ruff: noqa: F821

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

# Obtain forward references
if TYPE_CHECKING:
    from ._domains._array import Array
    from ._polys._poly import Poly

ElementLike = Union[int, str, "Array"]
"""
A :obj:`~typing.Union` representing objects that can be coerced into a Galois field element.

:group: arrays

Scalars are 0-D :obj:`~galois.Array` objects.

.. rubric:: Union

- :obj:`int`: A finite field element in its :ref:`integer representation <int-repr>`.

  .. ipython-with-reprs:: int,poly,power

      GF = galois.GF(3**5)
      GF(17)

- :obj:`str`: A finite field element in its :ref:`polynomial representation <poly-repr>`. Many string conventions are
  accepted, including: with/without `*`, with/without spaces, `^` or `**`, any indeterminate variable,
  increasing/decreasing degrees, etc. Or any combination of the above.

  .. ipython-with-reprs:: int,poly,power

      GF("x^2 + 2x + 2")

      # Add explicit * for multiplication
      GF("x^2 + 2*x + 2")

      # No spaces
      GF("x^2+2x+2")

      # ** instead of ^
      GF("x**2 + 2x + 2")

      # Different indeterminate
      GF("α^2 + 2α + 2")

      # Ascending degrees
      GF("2 + 2x + x^2")

- :obj:`~galois.Array`: A previously created scalar :obj:`~galois.Array` object. No coercion is necessary.

.. rubric:: Alias
"""


IterableLike = Union[Sequence[ElementLike], Sequence["IterableLike"]]
"""
A :obj:`~typing.Union` representing iterable objects that can be coerced into a Galois field array.

:group: arrays

.. rubric:: Union

- :obj:`~typing.Sequence` [ :obj:`~galois.typing.ElementLike` ]: An iterable of elements.

  .. ipython-with-reprs:: int,poly,power

      GF = galois.GF(3**5)
      GF([17, 4])
      # Mix and match integers and strings
      GF([17, "x + 1"])

- :obj:`~typing.Sequence` [ :obj:`~galois.typing.IterableLike` ]: A recursive iterable of iterables of elements.

  .. ipython-with-reprs:: int,poly,power

      GF = galois.GF(3**5)
      GF([[17, 4], [148, 205]])
      # Mix and match integers and strings
      GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])

.. rubric:: Alias
"""


ArrayLike = Union[IterableLike, np.ndarray, "Array"]
"""
A :obj:`~typing.Union` representing objects that can be coerced into a Galois field array.

:group: arrays

.. rubric:: Union

- :obj:`~galois.typing.IterableLike`: A recursive iterable of iterables of elements.

  .. ipython-with-reprs:: int,poly,power

      GF = galois.GF(3**5)
      GF([[17, 4], [148, 205]])
      # Mix and match integers and strings
      GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])

- :obj:`~numpy.ndarray`: A NumPy array of integers, representing finite field elements in their :ref:`integer
  representation <int-repr>`.

  .. ipython-with-reprs:: int,poly,power

      x = np.array([[17, 4], [148, 205]]); x
      GF(x)

- :obj:`~galois.Array`: A previously created :obj:`~galois.Array` object. No coercion is necessary.

.. rubric:: Alias
"""


ShapeLike = Union[int, Sequence[int]]
"""
A :obj:`~typing.Union` representing objects that can be coerced into a NumPy :obj:`~numpy.ndarray.shape` tuple.

:group: arrays

.. rubric:: Union

- :obj:`int`: The size of a 1-D array.

  .. ipython:: python

      GF = galois.GF(3**5)
      x = GF.Random(4); x
      x.shape

- :obj:`~typing.Sequence` [ :obj:`int` ]: An iterable of integer dimensions. Tuples or lists are allowed. An empty
  iterable, `()` or `[]`, represents a 0-D array (scalar).

  .. ipython:: python

      x = GF.Random((2, 3)); x
      x.shape
      x = GF.Random([2, 3, 4]); x
      x.shape
      x = GF.Random(()); x
      x.shape

.. rubric:: Alias
"""


DTypeLike = Union[np.integer, int, str, object]
"""
A :obj:`~typing.Union` representing objects that can be coerced into a NumPy data type.

:group: arrays

.. rubric:: Union

- :obj:`numpy.integer`: A fixed-width NumPy integer data type.

  .. ipython:: python

      GF = galois.GF(3**5)
      x = GF.Random(4, dtype=np.uint16); x.dtype
      x = GF.Random(4, dtype=np.int32); x.dtype

- :obj:`int`: The system default integer.

  .. ipython:: python

      x = GF.Random(4, dtype=int); x.dtype

- :obj:`str`: The string that can be coerced with :obj:`numpy.dtype`.

  .. ipython:: python

      x = GF.Random(4, dtype="uint16"); x.dtype
      x = GF.Random(4, dtype="int32"); x.dtype

- :obj:`object`: A Python object data type. This applies to non-compiled fields.

  .. ipython:: python

      GF = galois.GF(2**100)
      x = GF.Random(4, dtype=object); x.dtype

.. rubric:: Alias
"""


PolyLike = Union[int, str, ArrayLike, "Poly"]
"""
A :obj:`~typing.Union` representing objects that can be coerced into a polynomial.

:group: polys

.. rubric:: Union

- :obj:`int`: A polynomial in its integer representation, see :func:`~galois.Poly.Int`. The Galois field must be known
  from context.

  .. ipython:: python

      # Known from context
      GF = galois.GF(3)
      galois.Poly.Int(19, field=GF)

- :obj:`str`: A polynomial in its string representation, see :func:`~galois.Poly.Str`. The Galois field must be known
  from context.

  .. ipython:: python

      galois.Poly.Str("2x^2 + 1", field=GF)

- :obj:`~galois.typing.ArrayLike`: An array of polynomial coefficients in degree-descending order. If the coefficients
  are not :obj:`~galois.Array`, then the Galois field must be known from context.

  .. ipython:: python

      galois.Poly([2, 0, 1], field=GF)
      galois.Poly(GF([2, 0, 1]))

- :obj:`~galois.Poly`: A previously created :obj:`~galois.Poly` object. No coercion is necessary.

.. rubric:: Alias
"""


# Remove imported objects from public galois.typing namespace
del annotations
del Sequence, Union, TYPE_CHECKING
del np
