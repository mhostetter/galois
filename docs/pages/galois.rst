galois
======

A performant numpy extension for Galois fields and their applications.

.. currentmodule:: galois

Galois Fields
-------------

.. rubric:: Galois field class creation
.. autosummary::
   :template: class.rst
   :toctree:

   GF
   Field
   FieldArray
   FieldClass

.. rubric:: Pre-made Galois field classes
.. autosummary::
   :template: class_gf2.rst
   :toctree:

   GF2

.. rubric:: Prime field functions
.. autosummary::
   :toctree:

   primitive_root
   primitive_roots
   is_primitive_root

.. rubric:: Extension field functions
.. autosummary::
   :toctree:

   irreducible_poly
   irreducible_polys
   is_irreducible
   primitive_poly
   primitive_polys
   is_primitive
   conway_poly
   matlab_primitive_poly
   primitive_element
   primitive_elements
   is_primitive_element
   minimal_poly

.. rubric:: Galois fields for cryptography
.. autosummary::
   :template: class.rst
   :toctree:

   Oakley1
   Oakley2
   Oakley3
   Oakley4

Polynomials over Galois Fields
------------------------------

.. rubric:: Polynomial classes
.. autosummary::
   :template: class.rst
   :toctree:

   Poly

.. rubric:: Polynomial functions
.. autosummary::
   :toctree:

   poly_gcd
   poly_pow
   poly_factors

.. rubric:: Create specific polynomials
.. autosummary::
   :toctree:

   irreducible_poly
   irreducible_polys
   primitive_poly
   primitive_polys
   conway_poly
   matlab_primitive_poly
   minimal_poly

.. rubric:: Polynomial tests
.. autosummary::
   :toctree:

   is_monic
   is_irreducible
   is_primitive

Linear Sequences
----------------

.. rubric::
.. autosummary::
   :template: class.rst
   :toctree:

   berlekamp_massey

Forward Error Correcting Codes
------------------------------

.. rubric:: Code classes
.. autosummary::
   :template: class.rst
   :toctree:

   BCH
   ReedSolomon

.. rubric:: Cyclic code functions
.. autosummary::
   :template: class.rst
   :toctree:

   bch_valid_codes
   poly_to_generator_matrix
   roots_to_parity_check_matrix

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
   iroot
   ilog
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

.. rubric:: Prime numbers
.. autosummary::
   :toctree:

   primes
   kth_prime
   prev_prime
   next_prime
   random_prime
   mersenne_exponents
   mersenne_primes

.. rubric:: Primality tests
.. autosummary::
   :toctree:

   is_prime
   is_prime_fermat
   is_prime_miller_rabin

.. rubric:: Prime factorization
.. autosummary::
   :toctree:

   prime_factors
   is_smooth
