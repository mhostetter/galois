Galois Fields
=============

This section contains classes and functions for creating Galois field arrays.

.. currentmodule:: galois

Galois field classes
--------------------

Class factory functions
.......................

.. autosummary::

   GF
   Field

Abstract base classes
.....................

.. autosummary::
   :template: class.rst

   FieldArray

Pre-made :obj:`~galois.FieldArray` subclasses
.............................................

.. autosummary::
   :template: class.rst

   GF2

Prime field functions
---------------------

Primitive roots
...............

.. autosummary::

   primitive_root
   primitive_roots
   is_primitive_root

Extension field functions
-------------------------

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
   is_primitive

Primitive elements
..................

.. autosummary::

   primitive_element
   primitive_elements
   is_primitive_element
