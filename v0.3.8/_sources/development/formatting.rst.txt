Formatting
==========

The :obj:`galois` library uses `Ruff <https://docs.astral.sh/ruff/>`_ for static analysis, linting, and code
formatting.

Install
-------

First, `ruff` needs to be installed on your system. Easily install it by installing the development dependencies.

.. code-block:: console

   $ python3 -m pip install -r requirements-dev.txt

Configuration
-------------

The `ruff` configuration is provided in `pyproject.toml`.

.. literalinclude:: ../../pyproject.toml
   :caption: pyproject.toml
   :start-at: [tool.ruff]
   :end-before: [tool.pytest.ini_options]
   :language: toml

Run the linter
--------------

Run the Ruff linter manually from the command line.

.. code-block:: console

   $ python3 -m ruff check .

Run the formatter
-----------------

Run the Ruff formatter manually from the command line.

.. code-block:: console

   $ python3 -m ruff format --check .

Pre-commit
----------

A `pre-commit` configuration file with various hooks is provided in `.pre-commit-config.yaml`.

.. literalinclude:: ../../.pre-commit-config.yaml
   :caption: .pre-commit-config.yaml
   :language: yaml

Enable `pre-commit` by installing the pre-commit hooks.

.. code-block:: console

   $ pre-commit install

Run `pre-commit` on all files.

.. code-block:: console

   $ pre-commit run --all-files

Disable `pre-commit` by uninstalling the pre-commit hooks.

.. code-block:: console

   $ pre-commit uninstall

Run from VS Code
----------------

Install the `Ruff extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_ for VS Code.
Included is a VS Code configuration file `.vscode/settings.json`.
VS Code will run the linter and formatter as you view and edit files.
