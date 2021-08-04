{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block constructors %}
   {% if methods %}
   .. rubric:: {{ _('Constructors') }}

   .. autosummary::
   {% for item in methods if item[0].isupper() %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods if not item[0].isupper() %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block special_methods %}
   {% if methods %}
   .. rubric:: {{ _('Special Methods') }}

   .. autosummary::
   {% for item in ['__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__divmod__', '__mod__', '__pow__'] if item in members %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
