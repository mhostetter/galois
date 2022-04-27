"""
A performant NumPy extension for Galois fields and their applications.
"""
# pylint: disable=wrong-import-position

from ._version import __version__

# Nested modules
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

# Modules
from ._lfsr import *
from ._math import *
from ._modular import *
from ._ntt import *
from ._options import *
from ._polymorphic import *
from ._prime import *

# Subpackages
from . import typing
