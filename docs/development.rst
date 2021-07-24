Development
===========

For users who would like to actively develop with :obj:`galois`, these sections may prove helpful.

Lint the package
----------------

Linting is done with `pylint <https://www.pylint.org/>`_. The linting dependencies are stored in `requirements-lint.txt`.

.. literalinclude:: ../../requirements-lint.txt
   :caption: requirements-lint.txt
   :linenos:

Install the linter dependencies.

.. code-block:: sh

   $ python3 -m pip install -r requirements-lint.txt

Run the linter.

.. code-block:: sh

   $ python3 -m pylint --rcfile=setup.cfg galois/

Run the unit tests
------------------

Unit testing is done through `pytest <https://docs.pytest.org/en/stable/>`_. The tests themselves are stored in `tests/`. We test
against test vectors, stored in `tests/data/`. generated using `SageMath <https://www.sagemath.org/>`_.
See the `scripts/generate_test_vectors.py` script. The testing dependencies are stored in `requirements-test.txt`.

.. literalinclude:: ../../requirements-test.txt
   :caption: requirements-test.txt
   :linenos:

Install the test dependencies.

.. code-block:: sh

   $ python3 -m pip install -r requirements-test.txt

Run the unit tests.

.. code-block:: sh

   $ python3 -m pytest tests/

Build the documentation
-----------------------

The documentation is generated with `Sphinx <https://www.sphinx-doc.org/en/master/>`_. The dependencies are
stored in `requirements-doc.txt`.

.. literalinclude:: ../../requirements-doc.txt
   :caption: requirements-doc.txt
   :linenos:

Install the documentation dependencies.

.. code-block:: sh

   $ python3 -m pip install -r requirements-doc.txt

Build the HTML documentation. The index page will be located at `docs/build/index.html`.

.. code-block:: sh

   $ sphinx-build -b html -v docs/build/
