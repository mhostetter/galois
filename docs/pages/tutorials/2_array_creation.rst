Array creation
==============

Explicit construction
---------------------

Galois field arrays can be constructed either explicitly or through `numpy` view casting. The method of array
creation is the same for all Galois fields, but :math:`\mathrm{GF}(7)` is used as an example here.

.. ipython:: python

   # Represents an existing numpy array
   x_np = np.random.randint(0, 7, 10, dtype=int); x_np

   # Create a Galois field array through explicit construction (x_np is copied)
   x = GF7(x_np); x


View casting
------------

.. ipython:: python

   # View cast an existing array to a Galois field array (no copy operation)
   y = x_np.view(GF7); y

.. warning::

   View casting creates a pointer to the original data and simply interprets it as a new :obj:`numpy.ndarray` subclass,
   namely the Galois field classes. So, if the original array is modified so will the Galois field array.

   .. ipython:: python

      x_np

      # Add 1 (mod 7) to the first element of x_np
      x_np[0] = (x_np[0] + 1) % 7; x_np

      # Notice x is unchanged due to the copy during the explicit construction
      x

      # Notice y is changed due to view casting
      y


Alternate constructors
----------------------

There are alternate constructors for convenience: :obj:`galois.GFArray.Zeros`, :obj:`galois.GFArray.Ones`, :obj:`galois.GFArray.Range`,
:obj:`galois.GFArray.Random`, and :obj:`galois.GFArray.Elements`.

.. ipython:: python

   GF256.Random((2,5))
   GF256.Range(10,20)
   GF256.Elements()


Array dtypes
------------

Galois field arrays support all signed and unsigned integer dtypes, presuming the data type can store values
in :math:`[0, p^m)`. The default dtype is the smallest valid unsigned dtype.

.. ipython:: python

   GF = galois.GF(7)
   a = GF.Random(10); a
   a.dtype

   # Type cast an existing Galois field array to a different dtype
   a = a.astype(np.int16); a
   a.dtype

A specific dtype can be chosen by providing the `dtype` keyword argument during array creation.

.. ipython:: python

   # Explicitly create a Galois field array with a specific dtype
   b = GF.Random(10, dtype=np.int16); b
   b.dtype


Field element display modes
---------------------------

The default representation of a finite field element is the integer representation. That is, for :math:`\mathrm{GF}(p^m)`
the :math:`p^m` elements are represented as :math:`\{0,1,\dots,p^m-1\}`. For extension fields, the field elements can
alternatively be represented as polynomials in :math:`\mathrm{GF}(p)[x]` with degree less than :math:`m`. For prime fields, the integer
and polynomial representations are equivalent because in the polynomial representation each element is a degree-:math`0` polynomial over
:math:`\mathrm{GF}(p)`.

For example, in :math:`\mathrm{GF}(2^3)` the integer representation of the :math:`8` field elements is :math:`\{0, 1, 2, 3, 4, 5, 6, 7\}`
and the polynomial representation is :math:`\{0,\ 1,\ x,\ x+1,\ x^2,\ x^2+1,\ x^2+x,\ x^2+x+1\}`.

.. ipython:: python

   GF = galois.GF(2**3)
   a = GF.Random(10)

   # The default mode represents the field elements as integers
   a

   # The display mode can be set to "poly" mode
   GF.display("poly"); a

   # The display mode can be set to "power" mode
   GF.display("power"); a

   # Reset the display mode to the default
   GF.display(); a

The :obj:`galois.GFArray.display` method can be called as a context manager.

.. ipython:: python

   # The original display mode
   print(a)

   # The new display context
   with GF.display("poly"):
      print(a)

   with GF.display("power"):
      print(a)

   # Returns to the original display mode
   print(a)
