Array Creation
==============

This page discusses the multiple ways to create arrays over finite fields. For this discussion, we are working in
the finite field :math:`\mathrm{GF}(3^5)`.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         GF = galois.GF(3**5)
         print(GF)

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         GF = galois.GF(3**5, display="poly")
         print(GF)

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         GF = galois.GF(3**5, display="power")
         print(GF)

Create a new array
------------------

A :ref:`Galois field array` can be created from various *array-like* objects.

Valid *array-like* objects are: single values, iterables of values, iterables of iterables, or NumPy arrays. The
*values* can be either integers or polynomial strings.

Integers
........

The most standard input is an iterable of integers. For extension fields, an integer input is the :ref:`integer
representation <Integer representation>` of the polynomial field element.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF([17, 4])

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF([17, 4])

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF([17, 4])

Iterables of iterables are also supported.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF([[17, 4], [148, 205]])

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF([[17, 4], [148, 205]])

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF([[17, 4], [148, 205]])

Polynomial strings
..................

In addition to the integer representation, field elements may be expressed in their :ref:`polynomial representation <Polynomial representation>`
using strings.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF(["x^2 + 2x + 2", "x + 1"])

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF(["x^2 + 2x + 2", "x + 1"])

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF(["x^2 + 2x + 2", "x + 1"])

Many string conventions are accepted, including: with/without `*`, with/without spaces, `^` or `**`,
any indeterminate variable, increasing/decreasing degrees, etc. Or any combination of the above.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         # Add explicit * for multiplication
         GF(["x^2 + 2*x + 2", "x + 1"])
         # No spaces
         GF(["x^2+2x+2", "x+1"])
         # ** instead of ^
         GF(["x**2 + 2x + 2", "x + 1"])
         # Different indeterminate
         GF(["α^2 + 2α + 2", "α + 1"])
         # Ascending degrees
         GF(["2 + 2x + x^2", "1 + x"])

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         # Add explicit * for multiplication
         GF(["x^2 + 2*x + 2", "x + 1"])
         # No spaces
         GF(["x^2+2x+2", "x+1"])
         # ** instead of ^
         GF(["x**2 + 2x + 2", "x + 1"])
         # Different indeterminate
         GF(["α^2 + 2α + 2", "α + 1"])
         # Ascending degrees
         GF(["2 + 2x + x^2", "1 + x"])

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         # Add explicit * for multiplication
         GF(["x^2 + 2*x + 2", "x + 1"])
         # No spaces
         GF(["x^2+2x+2", "x+1"])
         # ** instead of ^
         GF(["x**2 + 2x + 2", "x + 1"])
         # Different indeterminate
         GF(["α^2 + 2α + 2", "α + 1"])
         # Ascending degrees
         GF(["2 + 2x + x^2", "1 + x"])

Integers and polynomial strings may be mixed and matched.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF(["x^2 + 2x + 2", 4])

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF(["x^2 + 2x + 2", 4])

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF(["x^2 + 2x + 2", 4])

Polynomial coefficients
.......................

Rather than strings, the polynomial coefficients may be passed into `GF`'s constructor as length-:math:`m` vectors using
the :func:`galois.FieldArray.Vector` classmethod.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF.Vector([[0, 0, 1, 2, 2], [0, 0, 0, 1, 1]])

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF.Vector([[0, 0, 1, 2, 2], [0, 0, 0, 1, 1]])

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF.Vector([[0, 0, 1, 2, 2], [0, 0, 0, 1, 1]])

The :func:`galois.FieldArray.vector` method is the opposite operation. It converts extension field elements from :math:`\mathrm{GF}(p^m)`
into length-:math:`m` vectors over :math:`\mathrm{GF}(p)`.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF([17, 4]).vector()

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF([17, 4]).vector()

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF([17, 4]).vector()

NumPy array
...........

An integer NumPy array may also be passed into `GF`. The default keyword argument `copy=True` of the :obj:`galois.FieldArray`
constructor will create a copy of the array.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         x_np = np.array([213, 167, 4, 214, 209], dtype=int); x_np
         x = GF(x_np); x
         # Modifying x does not modify x_np
         x[0] = 0; x_np

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         x_np = np.array([213, 167, 4, 214, 209], dtype=int); x_np
         x = GF(x_np); x
         # Modifying x does not modify x_np
         x[0] = 0; x_np

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         x_np = np.array([213, 167, 4, 214, 209], dtype=int); x_np
         x = GF(x_np); x
         # Modifying x does not modify x_np
         x[0] = 0; x_np

View an existing array
----------------------

Instead of creating a *Galois field array* explicitly, you can convert an existing NumPy array into a *Galois field array*
temporarily and work with it in-place.

Simply call `.view(GF)` to *view* the NumPy array as a *Galois field array*. When finished working in the
finite field, call `.view(np.ndarray)` to *view* it back to a NumPy array.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         x_np = np.array([213, 167, 4, 214, 209], dtype=int); x_np
         x = x_np.view(GF); x
         # Modifying x does modify x_np!
         x[0] = 0; x_np

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         x_np = np.array([213, 167, 4, 214, 209], dtype=int); x_np
         x = x_np.view(GF); x
         # Modifying x does modify x_np!
         x[0] = 0; x_np

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         x_np = np.array([213, 167, 4, 214, 209], dtype=int); x_np
         x = x_np.view(GF); x
         # Modifying x does modify x_np!
         x[0] = 0; x_np

Scalars
-------

A single finite field element (a scalar) is a 0-D *Galois field array*. They are created by passing a single
:ref:`array-like object <Create a new array>` to the *Galois field array class* `GF`'s constructor.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         a = GF(17); a
         a = GF("x^2 + 2x + 2"); a
         a = GF.Vector([0, 0, 1, 2, 2]); a
         a.ndim

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         a = GF(17); a
         a = GF("x^2 + 2x + 2"); a
         a = GF.Vector([0, 0, 1, 2, 2]); a
         a.ndim

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         a = GF(17); a
         a = GF("x^2 + 2x + 2"); a
         a = GF.Vector([0, 0, 1, 2, 2]); a
         a.ndim

Classmethods
------------

Several classmethods are provided in :obj:`galois.FieldArray` to assist with creating arrays.

Constant arrays
...............

The :func:`galois.FieldArray.Zeros` and :func:`galois.FieldArray.Ones` classmethods provide constant arrays that are
useful for initializing empty arrays.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF.Zeros(4)
         GF.Ones(4)

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF.Zeros(4)
         GF.Ones(4)

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF.Zeros(4)
         GF.Ones(4)

Ordered arrays
..............

The :func:`galois.FieldArray.Range` classmethod produces a range of elements similar to :func:`numpy.arange`. The integer `start`
and `stop` values are the :ref:`integer representation <Integer representation>` of the polynomial field elements.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF.Range(10, 20)

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF.Range(10, 20)

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF.Range(10, 20)

The :func:`galois.FieldArray.Elements` classmethod provides a 1-D array of all the finite field elements.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF.Elements()

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF.Elements()

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF.Elements()

Random arrays
.............

The :func:`galois.FieldArray.Random` classmethod provides a random array of the specified shape. This is convenient
for testing. The integer `low` and `high` values are the :ref:`integer representation <Integer representation>` of
the polynomial field elements.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF.Random(4, seed=1234)
         GF.Random(4, low=10, high=20, seed=5678)

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF.Random(4, seed=1234)
         GF.Random(4, low=10, high=20, seed=5678)

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF.Random(4, seed=1234)
         GF.Random(4, low=10, high=20, seed=5678)

Data types
----------

*Galois field arrays* support a fixed set of NumPy data types (:obj:`numpy.dtype`). The data type must be able to store
all the field elements (in their :ref:`integer representation <Integer representation>`).

Valid data types
................

For small finite fields, like :math:`\mathrm{GF}(2^4)`, every NumPy integer data type is supported.

.. ipython:: python

    GF = galois.GF(2**4)
    GF.dtypes

For medium finite fields, like :math:`\mathrm{GF}(2^{10})`, some NumPy integer data types are not supported. Here,
:obj:`numpy.uint8` and :obj:`numpy.int8` are not supported.

.. ipython:: python

    GF = galois.GF(2**10)
    GF.dtypes

For large finite fields, like :math:`\mathrm{GF}(2^{100})`, only the "object" data type (:obj:`numpy.object_`) is
supported. This uses arrays of Python objects, rather than integer data types. For *Galois field arrays*, the
Python objects used are Python integers, which have unlimited size.

.. ipython:: python

    GF = galois.GF(2**100)
    GF.dtypes

Default data type
.................

When arrays are created, unless otherwise specified, they use the default data type. The default data type is
the smallest unsigned data type (the first in the :obj:`galois.FieldClass.dtypes` list).

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

The data type may be explicitly set during array creation by setting the `dtype` keyword argument of the :obj:`galois.FieldArray`
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

..
   Reset the display mode to the integer representation so other pages aren't affected
.. ipython:: python
   :suppress:

   GF.display("int")
