Polynomials
===========

This section contains classes and functions for creating polynomials over Galois fields.

.. currentmodule:: galois

Polynomial classes
------------------

.. autosummary::
   :template: class.rst
   :toctree:

   Poly

Special polynomials
-------------------

Irreducible polynomials
.......................

.. autosummary::
   :toctree:

   irreducible_poly
   irreducible_polys

Primitive polynomials
.....................

.. autosummary::
   :toctree:

   primitive_poly
   primitive_polys
   conway_poly
   matlab_primitive_poly

Interpolating polynomials
.........................

.. autosummary::
   :toctree:

   lagrange_poly

Polynomial functions
--------------------

Divisibility
............

.. autosummary::
   :toctree:

   gcd
   egcd
   lcm
   prod
   are_coprime

Congruences
...........

.. autosummary::
   :toctree:

   crt

Factorization
..............

.. autosummary::
   :toctree:

   factors
   square_free_factorization
   distinct_degree_factorization
   equal_degree_factorization

Tests
.....

.. autosummary::
   :toctree:

   is_monic
   is_irreducible
   is_primitive
   is_square_free
