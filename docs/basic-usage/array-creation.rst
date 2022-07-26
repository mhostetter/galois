Array Creation
==============

This page discusses the multiple ways to create arrays over finite fields. For this discussion, we are working in
the finite field :math:`\mathrm{GF}(3^5)`.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         GF = galois.GF(3**5)
         print(GF.properties)

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         GF = galois.GF(3**5, display="poly")
         print(GF.properties)

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         GF = galois.GF(3**5, display="power")
         print(GF.properties)

Create a scalar
---------------

A single finite field element (a scalar) is a 0-D :obj:`~galois.FieldArray`. They are created by passing a single
:obj:`~galois.typing.ElementLike` object to `GF`'s constructor.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         a = GF(17); a
         a = GF("x^2 + 2x + 2"); a
         a.ndim

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         a = GF(17); a
         a = GF("x^2 + 2x + 2"); a
         a.ndim

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         a = GF(17); a
         a = GF("x^2 + 2x + 2"); a
         a.ndim

Create a new array
------------------

Array-Like objects
..................

A :obj:`~galois.FieldArray` can be created from various :obj:`~galois.typing.ArrayLike` objects.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF([17, 4, 148, 205])
         GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF([17, 4, 148, 205])
         GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF([17, 4, 148, 205])
         GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])

Polynomial coefficients
.......................

Rather than strings, the polynomial coefficients may be passed into `GF`'s constructor as length-:math:`m` vectors using
the :func:`~galois.FieldArray.Vector` classmethod.

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

The :func:`~galois.FieldArray.vector` method is the opposite operation. It converts extension field elements from :math:`\mathrm{GF}(p^m)`
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

Primitive element powers
........................

A :obj:`~galois.FieldArray` can also be created from the powers of a primitive element :math:`\alpha`.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         alpha = GF.primitive_element; alpha
         powers = np.array([222, 69, 54, 24]); powers
         alpha ** powers

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         alpha = GF.primitive_element; alpha
         powers = np.array([222, 69, 54, 24]); powers
         alpha ** powers

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         alpha = GF.primitive_element; alpha
         powers = np.array([222, 69, 54, 24]); powers
         alpha ** powers

NumPy array
...........

An integer NumPy array may also be passed into `GF`. The default keyword argument `copy=True` of the :obj:`~galois.FieldArray`
constructor will create a copy of the array.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         x_np = np.array([213, 167, 4, 214, 209]); x_np
         x = GF(x_np); x
         # Modifying x does not modify x_np
         x[0] = 0; x_np

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         x_np = np.array([213, 167, 4, 214, 209]); x_np
         x = GF(x_np); x
         # Modifying x does not modify x_np
         x[0] = 0; x_np

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
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

Classmethods
------------

Several classmethods are provided in :obj:`~galois.FieldArray` to assist with creating arrays.

Constant arrays
...............

The :func:`~galois.FieldArray.Zeros` and :func:`~galois.FieldArray.Ones` classmethods provide constant arrays that are
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

.. note::

   There is no :func:`numpy.empty` equivalent. This is because :obj:`~galois.FieldArray` instances must have values in
   :math:`[0, p^m)`. Empty NumPy arrays have whatever values are currently in memory, and therefore would fail those
   bounds checks.

Ordered arrays
..............

The :func:`~galois.FieldArray.Range` classmethod produces a range of elements similar to :func:`numpy.arange`. The integer `start`
and `stop` values are the :ref:`integer representation <int-repr>` of the polynomial field elements.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         @suppress
         GF.display("int")
         GF.Range(10, 20)
         GF.Range(10, 20, 2)

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         @suppress
         GF.display("poly")
         GF.Range(10, 20)
         GF.Range(10, 20, 2)

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         @suppress
         GF.display("power")
         GF.Range(10, 20)
         GF.Range(10, 20, 2)

Random arrays
.............

The :func:`~galois.FieldArray.Random` classmethod provides a random array of the specified shape. This is convenient
for testing. The integer `low` and `high` values are the :ref:`integer representation <int-repr>` of
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

..
   Reset the display mode to the integer representation so other pages aren't affected
.. ipython:: python
   :suppress:

   GF.display("int")

Class properties
----------------

Certain class properties, such as :obj:`~galois.FieldArray.elements`, :obj:`~galois.FieldArray.units`, :obj:`~galois.FieldArray.squares`,
and :obj:`~galois.FieldArray.primitive_elements`, provide an array of elements with the specified properties.

.. tab-set::

   .. tab-item:: Integer
      :sync: int

      .. ipython:: python

         GF = galois.GF(3**2)
         GF.elements
         GF.units
         GF.squares
         GF.primitive_elements

   .. tab-item:: Polynomial
      :sync: poly

      .. ipython:: python

         GF = galois.GF(3**2, display="poly")
         GF.elements
         GF.units
         GF.squares
         GF.primitive_elements

   .. tab-item:: Power
      :sync: power

      .. ipython:: python

         GF = galois.GF(3**2, display="power")
         GF.elements
         GF.units
         GF.squares
         GF.primitive_elements
         @suppress
         GF.display()

Data types
----------

:obj:`~galois.FieldArray` instances support a fixed set of NumPy data types (:obj:`numpy.dtype`). The data type must be
able to store all the field elements (in their :ref:`integer representation <int-repr>`).

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
