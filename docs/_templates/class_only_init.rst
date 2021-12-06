{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members: Elements, Identity, Ones, Random, Range, Vandermonde, Vector, Zeros

   {% block methods %}
   .. automethod:: __init__

   {% endblock %}
