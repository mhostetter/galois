"""
A performant numpy extension for Galois fields and their applications.
"""
from ._version import __version__

# Subpackages
from ._code import *
from ._field import *

# Modules
from ._factor import *
from ._integer import *
from ._lfsr import *
from ._log import *
from ._math import *  # pylint: disable=redefined-builtin
from ._modular import *
from ._prime import *
from ._structure import *
