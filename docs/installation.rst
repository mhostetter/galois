Installation
============

Install with pip
----------------

The latest released version of :obj:`galois` can be installed from `PyPI <https://pypi.org/project/galois/>`_ using `pip`.

.. code-block:: sh

   $ python3 - m pip install galois

.. note::

   Fun fact: read `here <https://snarky.ca/why-you-should-use-python-m-pip/>`_ from python core developer `Brett Cannon <https://twitter.com/brettsky>`_ about why it's better
   to install using `python3 -m pip` rather than `pip3`.

Install for development
-----------------------

The latest code from `master` can be checked out and installed locally in an `"editable" <https://pip.pypa.io/en/stable/cli/pip_install/?highlight=--editable#editable-installs>`_ fashion.
The "editable" install allows local changes to the `galois/` folder to be seen system-wide upon running `import galois`.

.. code-block:: sh

   $ git clone https://github.com/mhostetter/galois.git
   $ python3 -m pip install -e galois

Also, feel free to fork :obj:`galois` on GitHub, clone your fork, make changes, and contribute back with a pull request!

Install for development with min dependencies
---------------------------------------------

The package dependencies have minimum supported versions. They are stored in `requirements-min.txt`.

.. literalinclude:: ../requirements-min.txt
   :caption: requirements-min.txt
   :linenos:

`pip` installing :obj:`galois` will install the latest versions of the dependencies. If you'd like to test against
the oldest supported dependencies, you can do the following:

.. code-block:: sh

   $ git clone https://github.com/mhostetter/galois.git

   # First install the minimum version of the dependencies
   $ python3 -m pip install -r galois/requirements-min.txt

   # Then, installing the galois package won't upgrade the dependencies since their versions are satisfactory
   $ python3 -m pip install -e galois
