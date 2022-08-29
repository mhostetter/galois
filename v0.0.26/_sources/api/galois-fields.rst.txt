Galois Fields
=============

This section contains classes and functions for creating Galois field arrays.

.. currentmodule:: galois

Galois field classes
--------------------

Class factory functions
.......................

.. autosummary::
   :toctree:

   GF
   Field

Abstract base classes
.....................

.. autosummary::
   :template: class.rst
   :toctree:

   FieldArray
   FieldClass

Pre-made Galois field array classes
...................................

.. autosummary::
   :template: class_only_init.rst
   :toctree:

   GF2

Prime field functions
---------------------

Primitive roots
...............

.. autosummary::
   :toctree:

   primitive_root
   primitive_roots
   is_primitive_root

Extension field functions
-------------------------

Irreducible polynomials
.......................

.. autosummary::
   :toctree:

   irreducible_poly
   irreducible_polys
   is_irreducible

Primitive polynomials
.....................

.. autosummary::
   :toctree:

   primitive_poly
   primitive_polys
   conway_poly
   matlab_primitive_poly
   is_primitive

Primitive elements
..................

.. autosummary::
   :toctree:

   primitive_element
   primitive_elements
   is_primitive_element
