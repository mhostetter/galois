Documentation
=============

The :obj:`galois` documentation is generated with `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and the
`Sphinx Immaterial <https://jbms.github.io/sphinx-immaterial/>`_ theme.

Install
-------

The documentation dependencies are stored in `docs/requirements.txt`.

.. literalinclude:: ../requirements.txt
   :linenos:

Install the documentation dependencies by passing the `-r` switch to `pip install`.

.. code-block:: sh

   $ python3 -m pip install -r docs/requirements.txt

Build the docs
--------------

The documentation is built by running the `sphinx-build` command.

.. code-block:: sh

    $ sphinx-build -b dirhtml -v docs/ docs/build/

The HTML output is located in `docs/build/`. The home page is `docs/build/index.html`.
