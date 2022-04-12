"""
A performant NumPy extension for Galois fields and their applications.
"""
# pylint: disable=redefined-builtin,wrong-import-position

from ._version import __version__

# Subpackages
from ._polys import *  # Needs to be imported before _fields
from ._fields import *
###############################################################################
# Monkey-patch the default array type to be GF(2)
from . import _polys
setattr(_polys._poly, "DEFAULT_ARRAY", GF2)
###############################################################################
from ._codes import *  # Needs monkey patching before importing

# Modules
from ._array import *
from ._lfsr import *
from ._math import *
from ._modular import *
from ._ntt import *
from ._polymorphic import *
from ._prime import *
