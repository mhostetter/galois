{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   .. rubric:: {{ _('Constructors') }}
   .. autosummary::

      Elements
      Identity
      Ones
      Random
      Range
      Vandermonde
      Vector
      Zeros

   .. rubric:: {{ _('Methods') }}
   .. autosummary::

      lu_decompose
      lup_decompose
      row_reduce
      vector
