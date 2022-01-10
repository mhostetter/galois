"""
A performant NumPy extension for Galois fields and their applications.
"""
# pylint: disable=redefined-builtin

from ._version import __version__

# Subpackages
from ._codes import *
from ._fields import *

# Modules
from ._factor import *
from ._lfsr import *
from ._math import *
from ._modular import *
from ._ntt import *
from ._polymorphic import *
from ._prime import *
