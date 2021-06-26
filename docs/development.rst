Development
===========

For users who would like to actively develop with :obj:`galois`, these sections may prove helpful.

Install for development
-----------------------

The the latest code from `master` can be checked out and installed locally in an "editable" fashion.

.. code-block:: sh

   $ git clone https://github.com/mhostetter/galois.git
   $ python3 -m pip install -e galois


Install for development with min dependencies
---------------------------------------------

The package dependencies have minimum supported version. They are stored in `requirements-min.txt`.

.. literalinclude:: ../../requirements-min.txt
   :caption: requirements-min.txt
   :linenos:

`pip` installing :obj:`galois` will install the latest versions of the dependencies. If you'd like to test against
the oldest supported dependencies, you can do the following:

.. code-block:: sh

   $ git clone https://github.com/mhostetter/galois.git

   # First install the minimum version of the dependencies
   $ python3 -m pip install -r galois/requirements-min.txt

   # Then, installing the package won't upgrade the dependencies since their versions are satisfactory
   $ python3 -m pip install -e galois

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
