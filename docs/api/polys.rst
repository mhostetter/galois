Polynomials
===========

This section contains classes and functions for creating polynomials over Galois fields.

.. currentmodule:: galois

Polynomial classes
------------------

.. autosummary::
   :template: class.rst

   Poly

Special polynomials
-------------------

Irreducible polynomials
.......................

.. autosummary::

   irreducible_poly
   irreducible_polys

Primitive polynomials
.....................

.. autosummary::

   primitive_poly
   primitive_polys
   conway_poly
   matlab_primitive_poly

Interpolating polynomials
.........................

.. autosummary::

   lagrange_poly

Polynomial functions
--------------------

Divisibility
............

.. autosummary::

   gcd
   egcd
   lcm
   prod
   are_coprime

Congruences
...........

.. autosummary::

   crt

Factorization
..............

.. autosummary::

   factors
   square_free_factorization
   distinct_degree_factorization
   equal_degree_factorization

Tests
.....

.. autosummary::

   is_monic
   is_irreducible
   is_primitive
   is_square_free
