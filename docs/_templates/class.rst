{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {%- if methods %}
   {% set constructor_methods = ['__init__'] %}
   {%- for item in methods if item[0].isupper() %}
      {{- constructor_methods.append(item)|default("", True) }}
   {%- endfor %}

   {% set special_methods = [] %}
   {%- for item in methods if item.startswith('__') %}
     {{- special_methods.append(item)|default("", True) }}
   {%- endfor %}

   {% set regular_methods = [] %}
   {%- for item in methods if (not item[0].isupper() and not item.startswith('__')) %}
      {{- regular_methods.append(item)|default("", True) }}
   {%- endfor %}
   {% endif %}

   {% block constructor_methods %}
   {% if constructor_methods %}
   .. rubric:: {{ _('Constructors') }}
   .. autosummary::
   {% for item in constructor_methods %}
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

   {% block regular_methods %}
   {% if regular_methods %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::
   {% for item in regular_methods %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}
   .. autosummary::
   {% for item in attributes %}
      ~{{ objname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. automethod:: __init__
