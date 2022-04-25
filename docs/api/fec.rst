Forward Error Correction
========================

This section contains classes and functions for constructing forward error correction codes.

.. currentmodule:: galois

FEC classes
-----------

.. autosummary::
   :template: class.rst

   BCH
   ReedSolomon

Linear block code functions
---------------------------

.. autosummary::

   generator_to_parity_check_matrix
   parity_check_to_generator_matrix

Cyclic code functions
---------------------

.. autosummary::

   bch_valid_codes
   poly_to_generator_matrix
   roots_to_parity_check_matrix
