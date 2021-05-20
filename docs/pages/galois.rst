galois
======

A performant numpy extension for Galois fields.

.. currentmodule:: galois

Galois Fields
-------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   GF
   Field
   FieldArray
   FieldMeta
   GF2
   is_field
   is_prime_field
   is_extension_field

Prime Fields
------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   GF
   Field
   is_field
   is_prime_field
   is_prime
   primitive_root
   primitive_roots
   is_primitive_root

Extension Fields
----------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   GF
   Field
   is_field
   is_extension_field
   conway_poly
   irreducible_poly
   primitive_poly
   is_irreducible
   is_primitive
   primitive_element
   primitive_elements
   is_primitive_element
   minimal_poly

Galois Fields for Cryptography
------------------------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   Oakley1
   Oakley2
   Oakley3
   Oakley4

Polynomials over Galois Fields
------------------------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   Poly
   poly_gcd
   poly_pow
   poly_factors
   conway_poly
   irreducible_poly
   primitive_poly
   minimal_poly
   is_monic
   is_irreducible
   is_primitive

Finite Groups
-------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   Group
   GroupArray
   GroupMeta
   is_group
   is_cyclic
   euler_totient
   totatives

Modular Arithmetic
------------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   gcd
   lcm
   crt
   isqrt
   pow
   is_cyclic
   carmichael
   euler_totient
   totatives

Discrete Logarithms
-------------------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   log_naive

Primes
------
.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   primes
   kth_prime
   prev_prime
   next_prime
   random_prime
   mersenne_exponents
   mersenne_primes
   prime_factors
   is_smooth
   is_prime
   is_prime_fermat
   is_prime_miller_rabin
