{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   .. rubric:: {{ _('Galois Fields') }}

   .. autosummary::
      :toctree:

      GF
      GFArray
      GFMeta
      GF2

   .. rubric:: {{ _('Constructing Prime Fields') }}

   .. autosummary::
      :toctree:

      GF
      is_prime
      primitive_root
      primitive_roots
      is_primitive_root

   .. rubric:: {{ _('Constructing Extension Fields') }}

   .. autosummary::
      :toctree:

      GF
      conway_poly
      is_irreducible
      is_primitive
      primitive_element
      primitive_elements
      is_primitive_element

   .. rubric:: {{ _('Polynomials over Galois Fields') }}

   .. autosummary::
      :toctree:

      Poly
      poly_gcd
      poly_pow
      is_monic
      is_irreducible
      is_primitive

   .. rubric:: {{ _('Modular Arithmetic') }}

   .. autosummary::
      :toctree:

      gcd
      lcm
      crt
      isqrt
      is_cyclic
      carmichael
      euler_totient
      totatives

   .. rubric:: {{ _('Primes') }}

   .. autosummary::
      :toctree:

      primes
      kth_prime
      prev_prime
      next_prime
      random_prime
      prime_factors
      is_prime
      is_smooth
      fermat_primality_test
      miller_rabin_primality_test
