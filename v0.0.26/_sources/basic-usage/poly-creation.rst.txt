Polynomial Creation
===================

Univariate polynomials over finite fields are supported with the :obj:`galois.Poly` class.

Create a polynomial
-------------------

Create a polynomial by specifying its coefficients in degree-descending order and the finite field its over.

.. ipython:: python

   GF = galois.GF(2**8)
   galois.Poly([1, 0, 0, 55, 23], field=GF)

Or pass a *Galois field array* of coefficients without explicitly specifying the finite field.

.. ipython:: python

   coeffs = GF([1, 0, 0, 55, 23]); coeffs
   galois.Poly(coeffs)

Element representation
----------------------

As with *Galois field arrays*, the finite field element representation of the polynomial coefficients may be changed
by setting the `display` keyword argument of :func:`galois.GF` or using the :func:`galois.FieldClass.display` method.

.. ipython:: python

   GF = galois.GF(3**5)

   # Display f using the default integer representation
   f = galois.Poly([13, 0, 4, 2], field=GF); f

   # Display f using the polynomial representation
   GF.display("poly"); f

   # Display f using the power representation
   GF.display("power"); f

   GF.display("int");

See :ref:`Field Element Representation` for more details.

Classmethods
------------

There are several additional ways to create a polynomial. They are included as classmethods in :obj:`galois.Poly`.
By convention, classmethods use `PascalCase`, while methods use `snake_case`.

Alternate constructors
......................

Create a polynomial by specifying its non-zero degrees and coefficients using :func:`galois.Poly.Degrees`.

.. ipython:: python

   galois.Poly.Degrees([8, 1], coeffs=[1, 179], field=GF)

Create a polynomial from its integer representation using :func:`galois.Poly.Int`. Additionally, one may create a polynomial from
a binary, octal, or hexadecimal string of its integer representation.

.. tab-set::

   .. tab-item:: Integer

      .. ipython:: python

         galois.Poly.Int(268, field=GF)

   .. tab-item:: Binary string

      .. ipython:: python

         galois.Poly.Int(int("0b1011", 2))

   .. tab-item:: Octal string

      .. ipython:: python

         galois.Poly.Int(int("0o5034", 8), field=galois.GF(2**3))

   .. tab-item:: Hex string

      .. ipython:: python

         galois.Poly.Int(int("0xf700a275", 16), field=galois.GF(2**8))

Create a polynomial from its string representation using :func:`galois.Poly.Str`.

.. ipython:: python

   galois.Poly.Str("x^5 + 143", field=GF)

Create a polynomial from its roots using :func:`galois.Poly.Roots`.

.. ipython:: python

   f = galois.Poly.Roots([137, 22, 51], field=GF); f
   f.roots()

Simple polynomials
..................

The :func:`galois.Poly.Zero`, :func:`galois.Poly.One`, and :func:`galois.Poly.Identity` classmethods create common,
simple polynomials. They are included for convenience.

.. ipython:: python

   galois.Poly.Zero(GF)
   galois.Poly.One(GF)
   galois.Poly.Identity(GF)

Random polynomials
..................

Random polynomials of a given degree are easily created with :func:`galois.Poly.Random`.

.. ipython:: python

   galois.Poly.Random(4, field=GF)

Methods
-------

Polynomial objects have several methods that modify or perform operations on the polynomial. Below are some examples.

Compute the derivative of a polynomial using :func:`galois.Poly.derivative`.

.. ipython:: python

   GF = galois.GF(7)
   f = galois.Poly([1, 0, 5, 2, 3], field=GF); f
   f.derivative()

Compute the roots of a polynomial using :func:`galois.Poly.roots`.

.. ipython:: python

   f.roots()

Properties
----------

Polynomial objects have several instance properties. Below are some examples.

Find the non-zero degrees and coefficients of the polynomial using :obj:`galois.Poly.nonzero_degrees`
and :obj:`galois.Poly.nonzero_coeffs`.

.. ipython:: python

   GF = galois.GF(7)
   f = galois.Poly([1, 0, 3], field=GF); f
   f.nonzero_degrees
   f.nonzero_coeffs

Find the integer equivalent of the polynomial using :func:`int`, see :func:`galois.Poly.__int__`. Additionally, one may
convert a polynomial into the binary, octal, or hexadecimal string of its integer representation.

.. tab-set::

   .. tab-item:: Integer

      .. ipython:: python

         int(f)

   .. tab-item:: Binary string

      .. ipython:: python

         g = galois.Poly([1, 0, 1, 1]); g
         bin(g)

   .. tab-item:: Octal string

      .. ipython:: python

         g = galois.Poly([5, 0, 3, 4], field=galois.GF(2**3)); g
         oct(g)

   .. tab-item:: Hex string

      .. ipython:: python

         g = galois.Poly([0xf7, 0x00, 0xa2, 0x75], field=galois.GF(2**8)); g
         hex(g)

Get the string representation of the polynomial using :func:`str`.

.. ipython:: python

   str(f)

Special polynomials
-------------------

The :obj:`galois` library also includes several functions to find certain *special* polynomials. Below are some examples.

Find one or all irreducible polynomials with :func:`galois.irreducible_poly` and :func:`galois.irreducible_polys`.

.. ipython:: python

   galois.irreducible_poly(3, 3)
   list(galois.irreducible_polys(3, 3))

Find one or all primitive polynomials with :func:`galois.primitive_poly` and :func:`galois.primitive_polys`.

.. ipython:: python

   galois.primitive_poly(3, 3)
   list(galois.primitive_polys(3, 3))

Find the Conway polynomial using :func:`galois.conway_poly`.

.. ipython:: python

   galois.conway_poly(3, 3)
