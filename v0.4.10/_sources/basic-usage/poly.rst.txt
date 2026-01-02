Polynomials
===========

Univariate polynomials over finite fields are supported with the :obj:`~galois.Poly` class.

Create a polynomial
-------------------

Create a polynomial by specifying its coefficients in degree-descending order and the finite field its over.

.. ipython:: python

   GF = galois.GF(2**8)
   galois.Poly([1, 0, 0, 55, 23], field=GF)

Or pass a :obj:`~galois.FieldArray` of coefficients without explicitly specifying the finite field.

.. ipython:: python

   coeffs = GF([1, 0, 0, 55, 23]); coeffs
   galois.Poly(coeffs)

.. tip::
   :title: Use :func:`~galois.set_printoptions` to display the polynomial coefficients in degree-ascending order.
   :collapsible:

   .. ipython:: python

      galois.set_printoptions(coeffs="asc")
      galois.Poly(coeffs)
      @suppress
      galois.set_printoptions(coeffs="desc")

Element representation
----------------------

As with :obj:`~galois.FieldArray` instances, the finite field element representation of the polynomial coefficients may be changed
by setting the `repr` keyword argument of :func:`~galois.GF` or using the :func:`~galois.FieldArray.repr` classmethod.

.. ipython:: python

   GF = galois.GF(3**5)

   # Display f(x) using the default integer representation
   f = galois.Poly([13, 0, 4, 2], field=GF); print(f)

   # Display f(x) using the polynomial representation
   GF.repr("poly"); print(f)

   # Display f(x) using the power representation
   GF.repr("power"); print(f)

   GF.repr("int");

See :doc:`element-representation` for more details.

Alternate constructors
----------------------

There are several additional ways to create a polynomial. These alternate constructors are included as classmethods in :obj:`~galois.Poly`.
By convention, alternate constructors use `PascalCase` while other classmethods use `snake_case`.

Create a polynomial by specifying its non-zero degrees and coefficients using :func:`~galois.Poly.Degrees`.

.. ipython:: python

   galois.Poly.Degrees([1000, 1], coeffs=[1, 179], field=GF)

Create a polynomial from its integer representation using :func:`~galois.Poly.Int`. Additionally, one may create a polynomial from
a binary, octal, or hexadecimal string of its integer representation.

.. md-tab-set::

   .. md-tab-item:: Integer

      .. ipython:: python

         galois.Poly.Int(268, field=GF)

   .. md-tab-item:: Binary string

      .. ipython:: python

         galois.Poly.Int(int("0b1011", 2))

   .. md-tab-item:: Octal string

      .. ipython:: python

         galois.Poly.Int(int("0o5034", 8), field=galois.GF(2**3))

   .. md-tab-item:: Hex string

      .. ipython:: python

         galois.Poly.Int(int("0xf700a275", 16), field=galois.GF(2**8))

Create a polynomial from its string representation using :func:`~galois.Poly.Str`.

.. ipython:: python

   galois.Poly.Str("x^5 + 143", field=GF)

Create a polynomial from its roots using :func:`~galois.Poly.Roots`.

.. ipython:: python

   f = galois.Poly.Roots([137, 22, 51], field=GF); f
   f.roots()

The :func:`~galois.Poly.Zero`, :func:`~galois.Poly.One`, and :func:`~galois.Poly.Identity` classmethods create common,
simple polynomials. They are included for convenience.

.. ipython:: python

   galois.Poly.Zero(GF)
   galois.Poly.One(GF)
   galois.Poly.Identity(GF)

Random polynomials of a given degree are easily created with :func:`~galois.Poly.Random`.

.. ipython:: python

   galois.Poly.Random(4, field=GF)

Methods
-------

Polynomial objects have several methods that modify or perform operations on the polynomial. Below are some examples.

Compute the derivative of a polynomial using :func:`~galois.Poly.derivative`.

.. ipython:: python

   GF = galois.GF(7)
   f = galois.Poly([1, 0, 5, 2, 3], field=GF); f
   f.derivative()

Compute the roots of a polynomial using :func:`~galois.Poly.roots`.

.. ipython:: python

   f.roots()

Properties
----------

Polynomial objects have several instance properties. Below are some examples.

Find the non-zero degrees and coefficients of the polynomial using :obj:`~galois.Poly.nonzero_degrees`
and :obj:`~galois.Poly.nonzero_coeffs`.

.. ipython:: python

   GF = galois.GF(7)
   f = galois.Poly([1, 0, 3], field=GF); f
   f.nonzero_degrees
   f.nonzero_coeffs

Find the integer equivalent of the polynomial using `int()`, see :func:`~galois.Poly.__int__`. Additionally, one may
convert a polynomial into the binary, octal, or hexadecimal string of its integer representation.

.. md-tab-set::

   .. md-tab-item:: Integer

      .. ipython:: python

         int(f)

   .. md-tab-item:: Binary string

      .. ipython:: python

         g = galois.Poly([1, 0, 1, 1]); g
         bin(g)

   .. md-tab-item:: Octal string

      .. ipython:: python

         g = galois.Poly([5, 0, 3, 4], field=galois.GF(2**3)); g
         oct(g)

   .. md-tab-item:: Hex string

      .. ipython:: python

         g = galois.Poly([0xf7, 0x00, 0xa2, 0x75], field=galois.GF(2**8)); g
         hex(g)

Get the string representation of the polynomial using `str()`.

.. ipython:: python

   str(f)

Special polynomials
-------------------

The :obj:`galois` library also includes several functions to find certain *special* polynomials. Below are some examples.

Find one or all irreducible polynomials with :func:`~galois.irreducible_poly` and :func:`~galois.irreducible_polys`.

.. ipython:: python

   galois.irreducible_poly(3, 3)
   list(galois.irreducible_polys(3, 3))

Find one or all primitive polynomials with :func:`~galois.primitive_poly` and :func:`~galois.primitive_polys`.

.. ipython:: python

   galois.primitive_poly(3, 3)
   list(galois.primitive_polys(3, 3))

Find the Conway polynomial using :func:`~galois.conway_poly`.

.. ipython:: python

   galois.conway_poly(3, 3)
