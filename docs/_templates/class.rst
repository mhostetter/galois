{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {%- if methods %}
   {% set constructor_methods = [] %}
   {%- for item in ['__init__']  if item in members %}
     {{- constructor_methods.append(item)|default("", True) }}
   {%- endfor %}
   {%- for item in methods if item[0].isupper() %}
     {{- constructor_methods.append(item)|default("", True) }}
   {%- endfor %}

   {% set regular_methods = [] %}
   {%- for item in methods if not item[0].isupper() %}
     {{- regular_methods.append(item)|default("", True) }}
   {%- endfor %}

   {% set special_methods = [] %}
   {%- for item in ['__call__', '__len__', '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__divmod__', '__mod__', '__pow__'] if item in members %}
     {{- special_methods.append(item)|default("", True) }}
   {%- endfor %}
   {%- endif %}

   {% block constructor_methods %}
   {% if constructor_methods %}
   .. rubric:: {{ _('Constructors') }}

   .. autosummary::
   {% for item in constructor_methods %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block regular_methods %}
   {% if regular_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in regular_methods %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block special_methods %}
   {% if special_methods %}
   .. rubric:: {{ _('Special Methods') }}

   .. autosummary::
   {% for item in special_methods|sort %}
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
