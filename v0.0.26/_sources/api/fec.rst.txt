Forward Error Correction
========================

This section contains classes and functions for constructing forward error correction codes.

.. currentmodule:: galois

FEC classes
-----------

.. autosummary::
   :template: class.rst
   :toctree:

   BCH
   ReedSolomon

Linear block code functions
---------------------------

.. autosummary::
   :toctree:

   generator_to_parity_check_matrix
   parity_check_to_generator_matrix

Cyclic code functions
---------------------

.. autosummary::
   :toctree:

   bch_valid_codes
   poly_to_generator_matrix
   roots_to_parity_check_matrix
