Basic Usage
===========

Galois field array construction
-------------------------------

Construct Galois field array classes using the :obj:`galois.GF` class factory function.

.. ipython:: python

   import numpy as np
   import galois

   GF31 = galois.GF(31); print(GF31)
   issubclass(GF31, np.ndarray)

Galois field array classes contain extra class attributes related to the finite field.

.. ipython:: python

   # The size of the finite field
   GF31.order

   # A primitive element of the finite field
   GF31.primitive_element

   # The primitive polynomial of the finite field
   GF31.irreducible_poly

Create any Galois field array class type: `GF(2^m)`, `GF(p)`, or `GF(p^m)`. Even arbitrarily-large fields!

.. ipython:: python

   # Field used in AES
   GF256 = galois.GF(2**8); print(GF256)

   prime = 36893488147419103183; galois.is_prime(prime)
   # Large prime field
   GFp = galois.GF(prime); print(GFp)

   # Large characteristic-2 field
   GF2_100 = galois.GF(2**100); print(GF2_100)

Create arrays from existing `numpy` arrays.

.. ipython:: python

   # Represents an existing numpy array
   array = np.random.randint(0, GF256.order, 10, dtype=int); array

   # Explicit Galois field construction
   GF256(array)

   # Numpy view casting to a Galois field
   array.view(GF256)

Or, create Galois field arrays using alternate constructors.

.. ipython:: python

   x = GF256.Random(10); x

   # Construct a random array without zeros to prevent ZeroDivisonError
   y = GF256.Random(10, low=1); y

The class factory :obj:`galois.GF` stores flyweights of previously generated classes. So, after creating a Galois field array
class you can assign it to a variable for future use or always use class factory. Both are equally fast. It's up to personal
preference and coding style. When creating a Galois field array class with several keyword arguments, that may be a good time
to save the class to a variable.

.. ipython:: python

   # Create a Galois field array class and assign it to a variable
   GF256 = galois.GF(2**8); print(GF256)

   # Use the class variable to create arrays
   GF256.Random((2,5))

   # Or simply call the class factory each time
   galois.GF(2**8).Random((2,5))

Galois field array arithmetic
-----------------------------

Galois field arrays support traditional numpy array operations.

.. ipython:: python

   x + y

   -x

   x * y

   x / y

   # Multiple addition of a Galois field array with any integer
   x * -3  # NOTE: -3 is outside the field

   # Exponentiate a Galois field array with any integer
   y ** -2  # NOTE: -2 is outside the field

   # Log base alpha (the field's primitive element)
   np.log(y)

Even field arithmetic on extremely large fields!

.. ipython:: python

   m = GFp.Random(3)
   n = GFp.Random(3)
   m + n
   m ** 123456

   r = GF2_100.Random(3); r

   # With characteristic 2, this will always be zero
   r + r

   # This is equivalent
   r * 2

   # But this will result in `r`
   r * 3

Galois field arrays support numpy array broadcasting.

.. ipython:: python

   a = GF31.Random((2,5)); a
   b = GF31.Random(5); b

   a + b

Galois field arrays also support numpy ufunc methods.

.. ipython:: python

   # Valid ufunc methods include "reduce", "accumulate", "reduceat", "outer", "at"
   np.multiply.reduce(a, axis=0)

   np.multiply.outer(x, y)

Display field elements as integers or polynomials.

.. ipython:: python

   print(x)

   # Temporarily set the display mode to represent GF(p^m) field elements as polynomials over GF(p)[x].
   with GF256.display("poly"):
      print(x)

Galois field polynomial construction
------------------------------------

Construct Galois field polynomials.

.. ipython:: python

   # Construct a polynomial by specifying all the coefficients in descending-degree order
   p = galois.Poly([1, 22, 0, 17, 25], field=GF31); p

   # Construct a polynomial by specifying only the non-zero coefficients
   q = galois.Poly.Degrees([2, 0], coeffs=[4, 14], field=GF31); q

Galois field polynomial arithmetic
----------------------------------

Galois field polynomial arithmetic is similar to numpy array operations.

.. ipython:: python

   p + q
   p // q, p % q
   p ** 2

Galois field polynomials can also be evaluated at constants or arrays.

.. ipython:: python

   p
   a

   # Evaluate a polynomial at a single value
   p(1)

   # Evaluate a polynomial at an array of values
   p(a)
