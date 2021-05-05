{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block factory_functions %}
   {% if functions %}
   .. rubric:: {{ _('Class Factory Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {% if item in ["GF",] %}
         {{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block abstract_classes %}
   {% if classes %}
   .. rubric:: {{ _('Abstract Classes') }}

   .. autosummary::
      :template: class.rst
      :toctree:
   {% for item in classes %}
      {% if item in ["GFMeta", "GFArray"] %}
         {{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :template: class.rst
      :toctree:
   {% for item in classes %}
      {% if item not in ["GFMeta", "GFArray"] %}
         {{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {% if item not in ["GF",] %}
         {{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
