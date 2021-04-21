{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   .. rubric:: {{ _('General') }}

   .. autosummary::
      :toctree:

      np.copy
      np.concatenate
      np.insert

   .. rubric:: {{ _('Arithmetic') }}

   .. autosummary::
      :toctree:

      np.add
      np.subtract
      np.multiply
      np.divide
      np.negative
      np.reciprocal
      np.power
      np.square
      np.log
      np.matmul

   .. rubric:: {{ _('Linear Algebra') }}

   .. autosummary::
      :toctree:

      np.trace
      np.matmul
      np.linalg.matrix_rank
      np.linalg.matrix_power
      np.linalg.det
      np.linalg.inv
      np.linalg.solve
