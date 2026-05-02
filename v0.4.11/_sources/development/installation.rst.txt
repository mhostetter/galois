Installation
============

Install from PyPI
-----------------

The latest released version of :obj:`galois` can be installed from `PyPI <https://pypi.org/project/galois/>`_ using `pip`.

.. code-block:: console

   $ python3 -m pip install galois

Install from GitHub
-------------------

The latest code on `main` can be installed using `pip` in this way.

.. code-block:: console

   $ python3 -m pip install git+https://github.com/mhostetter/galois.git

Or from a specific branch.

.. code-block:: console

   $ python3 -m pip install git+https://github.com/mhostetter/galois.git@branch

Editable install from local folder
----------------------------------

To actively develop the library, it is beneficial to `pip install` the library in an `editable <https://pip.pypa.io/en/stable/cli/pip_install/?highlight=--editable#editable-installs>`_
fashion from a local folder. This allows any changes in the current directory to be immediately seen upon the next `import galois`.

Clone the repo wherever you'd like.

.. code-block:: console

    $ git clone https://github.com/mhostetter/galois.git

Install the local folder using the `-e` or `--editable` flag.

.. code-block:: console

    $ python3 -m pip install -e galois/

Install the development dependencies
------------------------------------

The development dependencies include packages for linting and unit testing.
These dependencies are stored in `requirements-lint.txt` and `requirements-test.txt`.

.. literalinclude:: ../../requirements-lint.txt
   :caption: requirements-lint.txt

.. literalinclude:: ../../requirements-test.txt
   :caption: requirements-test.txt

Install the development dependencies by passing `-r` to `pip install`.

.. code-block:: console

   $ python3 -m pip install -r requirements-lint.txt
   $ python3 -m pip install -r requirements-test.txt
