Development
===========

For users who would like to actively develop with :obj:`galois`, these sections may prove helpful.

Install the `dev` dependencies
------------------------------

The development dependencies include packages for linting and testing the package. These dependencies are stored
in `requirements-dev.txt`.

.. literalinclude:: ../requirements-dev.txt
   :caption: requirements-dev.txt
   :linenos:

Install the `dev` dependencies.

.. code-block:: sh

   $ python3 -m pip install -r requirements-dev.txt

Lint the package
----------------

Linting is done with `pylint <https://www.pylint.org/>`_. The linter can be run from the command line as follows. There is also a
`.vscode/` folder with appropriate settings, if using `VS Code <https://code.visualstudio.com/>`_.

.. code-block:: sh

   $ python3 -m pylint --rcfile=setup.cfg galois/

Run the unit tests
------------------

Unit testing is done with `pytest <https://docs.pytest.org/en/stable/>`_. The tests themselves are stored in `tests/`. We utilize
test vectors stored in `tests/data/`. The tests can be run from the command line as follows. There is also a `.vscode/` folder
with appropriate settings, if using `VS Code <https://code.visualstudio.com/>`_.

.. code-block:: sh

   $ python3 -m pytest tests/

Build the documentation
-----------------------

The documentation is generated with `Sphinx <https://www.sphinx-doc.org/en/master/>`_. The documentation dependencies are
stored in `docs/requirements.txt`.

.. literalinclude:: requirements.txt
   :caption: docs/requirements.txt
   :linenos:

Install the documentation dependencies.

.. code-block:: sh

   $ python3 -m pip install -r docs/requirements.txt

Build the HTML documentation. The index page will be located at `docs/build/index.html`.

.. code-block:: sh

   $ sphinx-build -b html -v docs/ docs/build/
