Constructing finite field arrays
================================


GF(2) field class
-----------------

The :math:`\mathrm{GF}(2)` field is already constructed in :obj:`galois`. It can be accessed by :obj:`galois.GF2`.

.. ipython:: python

   GF2 = galois.GF2
   print(GF2)

   # The primitive element of the finite field
   GF2.alpha

   # The primitive polynomial of the finite field
   GF2.prim_poly

   # The primitive element is a root of the primitive polynomial
   GF2.prim_poly(GF2.alpha)


GF(p) field classes
-------------------

:math:`\mathrm{GF}(p)` fields, where :math:`p` is prime, can be constructed using the class factory
:obj:`galois.GF_factory`.

.. ipython:: python

   GF7 = galois.GF_factory(7, 1)
   print(GF7)

   # The primitive element of the finite field
   GF7.alpha

   # The primitive polynomial of the finite field
   GF7.prim_poly

   # The primitive element is a root of the primitive polynomial
   GF7.prim_poly(GF7.alpha)


GF(2^m) field classes
---------------------

:math:`\mathrm{GF}(2^m)` fields, where :math:`m` is a positive integer, can be constructed using the class
factory :obj:`galois.GF_factory`.

.. ipython:: python

   GF8 = galois.GF_factory(2, 3)
   print(GF8)

   # The primitive element of the finite field
   GF8.alpha

   # The primitive polynomial of the finite field
   GF8.prim_poly

   # Convert the polynomial from GF(2)[x] to GF(8)[x]
   prim_poly = galois.Poly(GF8.prim_poly.coeffs, field=GF8); prim_poly

   # The primitive element is a root of the primitive polynomial in GF(8)
   prim_poly(GF8.alpha)


Array creation: explicit and view casting
-----------------------------------------

Galois field arrays can be constructed either explicitly or through `numpy` view casting. The method of array
creation is the same for all Galois fields, but :math:`\mathrm{GF}(7)` is used as an example here.

.. ipython:: python

   # Represents an existing numpy array
   x_np = np.random.randint(0, 7, 10, dtype=int); x_np

   # Create a Galois field array through explicit construction (x_np is copied)
   x = GF7(x_np); x

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


Galois field array dtypes
-------------------------

Galois field arrays support all signed and unsigned integer dtypes, presuming the data type can store values
in :math:`[0, p^m)`.

.. ipython:: python

   GF = galois.GF_factory(7, 1)
   a = GF.Random(10); a

   # Type cast an existing Galois field array to a different dtype
   a.astype(np.uint8)

   # Explicitly create a Galois field array with a specific dtype
   b = GF.Random(10, dtype=np.uint8); b


Field element display modes
---------------------------

The default representation of a finite field element is the integer representation. That is, for :math:`\mathrm{GF}(2^3)`
the 8 field elements are represented as :math:`\{0, 1, 2, 3, 4, 5, 6, 7\}`. Alternatively, the field elements can be represented
as degree-3 polynomials in :math:`\mathrm{GF}(2)[x]`, i.e. :math:`\{0, 1, x, x+1, x^2, x^2+1, x^2+x, x^2+x+1\}`.

.. ipython:: python

   GF = galois.GF_factory(2, 3)
   a = GF.Random(10)

   # The default mode represents the field elements as integers
   a

   # The display mode can be set to "poly" mode
   GF.display("poly"); a

   # The polynomial variable can also be set
   GF.display("poly", "r"); a

   # Reset the display mode to the default
   GF.display(); a
