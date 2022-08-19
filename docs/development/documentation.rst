Documentation
=============

The :obj:`galois` documentation is generated with `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and the
`Sphinx Immaterial <https://jbms.github.io/sphinx-immaterial/>`_ theme.

Install
-------

The documentation dependencies are stored in `pyproject.toml`.

.. literalinclude:: ../../pyproject.toml
   :caption: pyproject.toml
   :start-at: [project.optional-dependencies]
   :end-before: [project.urls]
   :language: toml

Install the documentation dependencies by passing the `[doc]` extras to `pip install`.

.. code-block:: sh

   $ python3 -m pip install .[doc]

Build the docs
--------------

The documentation is built by running the `sphinx-build` command.

.. code-block:: sh

   $ sphinx-build -b dirhtml -v docs/ docs/build/

The HTML output is located in `docs/build/`. The home page is `docs/build/index.html`.

Serve the docs
--------------

Since the site is built to use directories (`*/getting-started/` not `*/getting-started.html`), it is necessary
to serve the webpages locally with a webserver. This is easily done using the built-in Python `http` module.

.. code-block:: sh

   $ python3 -m http.server 8080 -d docs/build/

The documentation is accessible from a web browser at `http://localhost:8080/`.
