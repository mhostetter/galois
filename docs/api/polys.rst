Polynomials over Galois Fields
==============================

This section contains classes and functions for creating polynomials over Galois fields.

.. currentmodule:: galois

Polynomial classes
------------------

.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   Poly

Special polynomials
-------------------

.. rubric:: Irreducible polynomials
.. autosummary::
   :toctree:

   irreducible_poly
   irreducible_polys

.. rubric:: Primitive polynomials
.. autosummary::
   :toctree:

   primitive_poly
   primitive_polys
   conway_poly
   matlab_primitive_poly

.. rubric:: Interpolating polynomials
.. autosummary::
   :toctree:

   lagrange_poly

Polynomial functions
--------------------

.. rubric::  Divisibility
.. autosummary::
   :toctree:

   gcd
   egcd
   lcm
   prod
   are_coprime

.. rubric::  Congruences
.. autosummary::
   :toctree:

   pow
   crt

.. rubric::  Polynomial factorization
.. autosummary::
   :toctree:

   factors
   square_free_factorization
   distinct_degree_factorization
   equal_degree_factorization

.. rubric:: Polynomial tests
.. autosummary::
   :toctree:

   is_monic
   is_irreducible
   is_primitive
   is_square_free
