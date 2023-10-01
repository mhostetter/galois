Array Creation
==============

This page discusses the multiple ways to create arrays over finite fields. For this discussion, we are working in
the finite field $\mathrm{GF}(3^5)$.

.. ipython-with-reprs:: int,poly,power

   GF = galois.GF(3**5)
   print(GF.properties)
   alpha = GF.primitive_element; alpha

Create a scalar
---------------

A single finite field element (a scalar) is a 0-D :obj:`~galois.FieldArray`. They are created by passing a single
:obj:`~galois.typing.ElementLike` object to `GF`'s constructor. A finite field scalar may also be created by exponentiating
the primitive element to a scalar power.

.. ipython-with-reprs:: int,poly,power

   GF(17)
   GF("x^2 + 2x + 2")
   alpha ** 222

Create a new array
------------------

Array-like objects
..................

A :obj:`~galois.FieldArray` can be created from various :obj:`~galois.typing.ArrayLike` objects.
A finite field array may also be created by exponentiating the primitive element to a an array of powers.

.. ipython-with-reprs:: int,poly,power

   GF([17, 4, 148, 205])
   GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])
   alpha ** np.array([[222, 69], [54, 24]])

Polynomial coefficients
.......................

Rather than strings, the polynomial coefficients may be passed into `GF`'s constructor as length-$m$ vectors using
the :func:`~galois.FieldArray.Vector` classmethod.

.. ipython-with-reprs:: int,poly,power

   GF.Vector([[0, 0, 1, 2, 2], [0, 0, 0, 1, 1]])

The :func:`~galois.FieldArray.vector` method is the opposite operation. It converts extension field elements from $\mathrm{GF}(p^m)$
into length-$m$ vectors over $\mathrm{GF}(p)$.

.. ipython-with-reprs:: int,poly,power

   GF([17, 4]).vector()

NumPy array
...........

An integer NumPy array may also be passed into `GF`. The default keyword argument `copy=True` of the :obj:`~galois.FieldArray`
constructor will create a copy of the array.

.. ipython-with-reprs:: int,poly,power

   x_np = np.array([213, 167, 4, 214, 209]); x_np
   x = GF(x_np); x
   # Modifying x does not modify x_np
   x[0] = 0; x_np

View an existing array
----------------------

Instead of creating a :obj:`~galois.FieldArray` explicitly, you can convert an existing NumPy array into a :obj:`~galois.FieldArray`
temporarily and work with it in-place.

Simply call `.view(GF)` to *view* the NumPy array as a :obj:`~galois.FieldArray`. When finished working in the
finite field, call `.view(np.ndarray)` to *view* it back to a NumPy array.

.. ipython-with-reprs:: int,poly,power

   x_np = np.array([213, 167, 4, 214, 209], dtype=int); x_np
   x = x_np.view(GF); x
   # Modifying x does modify x_np!
   x[0] = 0; x_np

Classmethods
------------

Several classmethods are provided in :obj:`~galois.FieldArray` to assist with creating arrays.

Constant arrays
...............

The :func:`~galois.FieldArray.Zeros` and :func:`~galois.FieldArray.Ones` classmethods provide constant arrays that are
useful for initializing empty arrays.

.. ipython-with-reprs:: int,poly,power

   GF.Zeros(4)
   GF.Ones(4)

.. info::
   :title: There is no :func:`numpy.empty` equivalent.
   :collapsible:

   This is because :obj:`~galois.FieldArray` instances must have values in $[0, p^m)$. Empty NumPy arrays have whatever values
   are currently in memory, and therefore would fail those bounds checks during instantiation.

Ordered arrays
..............

The :func:`~galois.FieldArray.Range` classmethod produces a range of elements similar to :func:`numpy.arange`. The integer `start`
and `stop` values are the :ref:`integer representation <int-repr>` of the polynomial field elements.

.. ipython-with-reprs:: int,poly,power

   GF.Range(10, 20)
   GF.Range(10, 20, 2)

Random arrays
.............

The :func:`~galois.FieldArray.Random` classmethod provides a random array of the specified shape. This is convenient
for testing. The integer `low` and `high` values are the :ref:`integer representation <int-repr>` of
the polynomial field elements.

.. ipython-with-reprs:: int,poly,power

   GF.Random(4, seed=1)
   GF.Random(4, low=10, high=20, seed=2)

Class properties
----------------

Certain class properties, such as :obj:`~galois.FieldArray.elements`, :obj:`~galois.FieldArray.units`, :obj:`~galois.FieldArray.squares`,
and :obj:`~galois.FieldArray.primitive_elements`, provide an array of elements with the specified properties.

.. ipython-with-reprs:: int,poly,power

   GF = galois.GF(3**2)
   GF.elements
   GF.units
   GF.squares
   GF.primitive_elements

Data types
----------

:obj:`~galois.FieldArray` instances support a fixed set of NumPy data types (:obj:`numpy.dtype`). The data type must be
able to store all the field elements (in their :ref:`integer representation <int-repr>`).

Valid data types
................

For small finite fields, like $\mathrm{GF}(2^4)$, every NumPy integer data type is supported.

.. ipython:: python

    GF = galois.GF(2**4)
    GF.dtypes

For medium finite fields, like $\mathrm{GF}(2^{10})$, some NumPy integer data types are not supported. Here,
:obj:`numpy.uint8` and :obj:`numpy.int8` are not supported.

.. ipython:: python

    GF = galois.GF(2**10)
    GF.dtypes

For large finite fields, like $\mathrm{GF}(2^{100})$, only the "object" data type (:obj:`numpy.object_`) is
supported. This uses arrays of Python objects, rather than integer data types. The Python objects used are Python integers,
which have unlimited size.

.. ipython:: python

    GF = galois.GF(2**100)
    GF.dtypes

Default data type
.................

When arrays are created, unless otherwise specified, they use the default data type. The default data type is
the smallest unsigned data type (the first in the :obj:`~galois.FieldArray.dtypes` list).

.. ipython:: python

    GF = galois.GF(2**10)
    GF.dtypes
    x = GF.Random(4); x
    x.dtype

.. ipython:: python

    GF = galois.GF(2**100)
    GF.dtypes
    x = GF.Random(4); x
    x.dtype

Changing data types
...................

The data type may be explicitly set during array creation by setting the `dtype` keyword argument of the :obj:`~galois.FieldArray`
constructor.

.. ipython:: python

    GF = galois.GF(2**10)
    x = GF([273, 388, 124, 400], dtype=np.uint32); x
    x.dtype

Arrays may also have their data types changed using `.astype()`. The data type must be valid, however.

.. ipython:: python

    x.dtype
    x = x.astype(np.int64)
    x.dtype

NumPy functions
---------------

Most native NumPy functions work on :obj:`~galois.FieldArray` instances as expected. For example, reshaping a `(10,)`-shape array
into a `(2, 5)`-shape array works as desired and returns a :obj:`~galois.FieldArray` instance.

.. ipython:: python

   GF = galois.GF(7)
   x = GF.Random(10, seed=1); x
   np.reshape(x, (2, 5))

However, some functions have a `subok` keyword argument. This indicates whether to return a :obj:`numpy.ndarray` subclass
from the function. Most notably, :func:`numpy.copy` defaults `subok` to `False`.

.. ipython:: python

   x
   # Returns np.ndarray!
   np.copy(x)
   np.copy(x, subok=True)

The :func:`numpy.ndarray.copy` method will, however, return a subclass. Be mindful of the `subok` keyword argument!

.. ipython:: python

   x.copy()
