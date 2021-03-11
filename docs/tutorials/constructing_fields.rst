Constructing finite field arrays
================================


GF(2) field class
-----------------

The :math:`GF(2)` field is already constructed in `galois`. It can be accessed by `galois.GF2`.

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

:math:`GF(p)` fields, where :math:`p` is prime, can be constructed using the class factory
`galois.GF_factory(p, 1)`.

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

:math:`GF(2^m)` fields, where :math:`m` is a positive integer, can be constructed using the class
factory `galois.GF_factory(2, m)`.

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


Array creation, explicit and view casting
-----------------------------------------

Galois field arrays can be constructed either explicitly or through `numpy` view casting. The method of array
creation is the same for all Galois fields, but :math:`GF(7)` is used as an example here.

.. ipython:: python

   # Represents an existing numpy array
   x_np = np.random.randint(0, 7, 10, dtype=int); x_np

   # Create a Galois field array through explicit construction (x_np is copied)
   x = GF7(x_np); x

   # View cast an existing array to a Galois field array (no copy operation)
   y = x_np.view(GF7); y

.. warning::

   View casting creates a pointer to the original data and simply interprets it as a new `np.ndarray` subclass,
   namely the Galois field classes. So, if the original array is modified so will the Galois field array.

   .. ipython:: python

      x_np

      # Flip the LSB in the first element of x_np
      x_np[0] ^= 1; x_np

      # Notice x is unchanged due to the copy during the explicit construction
      x

      # Notice y is changed due to view casting
      y


Galois field array dtypes
-------------------------

Galois field arrays support all integer dtypes, presuming the data type.
