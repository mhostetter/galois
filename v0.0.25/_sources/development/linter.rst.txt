Linter
======

The :obj:`galois` library uses `pylint <https://pylint.org/>`_ for static analysis and code
formatting.

Install
-------

First, `pylint` needs to be installed on your system. Easily install it by installing the development dependencies.

.. code-block:: sh

   $ python3 -m pip install -r requirements-dev.txt

Configuration
-------------

Various nuisance `pylint` warnings are added to an ignore list in `setup.cfg`.

.. code-block:: ini
   :linenos:

    [pylint]
    disable =
        line-too-long,
        too-many-lines,
        # ...

Run from the command line
-------------------------

Run the linter manually from the command line by passing in the `setup.cfg` file as the `pylint` configuration file.

.. code-block:: sh

    $ python3 -m pylint --rcfile=setup.cfg galois/

Run from VS Code
----------------

Included is a VS Code configuration file `.vscode/settings.json`. This instructs VS Code about how to invoke `pylint`.
VS Code will run the linter as you view and edit files.
