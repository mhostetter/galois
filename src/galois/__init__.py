"""
A performant NumPy extension for Galois fields and their applications.
"""
# ruff: noqa: F405, E402
# isort: skip_file

try:
    from ._version import __version__, __version_tuple__
except ModuleNotFoundError:  # pragma: no cover
    import warnings

    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)
    warnings.warn(
        "An error occurred during package install where setuptools_scm failed to create a _version.py file."
        "Defaulting version to 0.0.0.",
        stacklevel=3,
    )


# Import class/functions from nested private modules
from ._domains import *
from ._polys import *  # Needs to be imported before _fields
from ._fields import *

###############################################################################
# Monkey-patch the finite field default array and class factory into the
# _polys module. This is needed here due to circular dependencies.
from ._domains import _factory

_factory.FIELD_FACTORY = GF
_factory.DEFAULT_ARRAY = GF2
###############################################################################
from ._codes import *  # Needs monkey patching before importing

# Import class/functions from private modules
from ._lfsr import *
from ._math import *
from ._modular import *
from ._ntt import *
from ._options import *
from ._polymorphic import *
from ._prime import *

# Import public modules
from . import typing
