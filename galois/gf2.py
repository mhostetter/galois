import types

from .array import GFArray
from .meta_gf2 import GF2Meta

# Create GF2 class for use in poly.py
GF2 = types.new_class("GF2", bases=(GFArray,), kwds={
    "metaclass": GF2Meta,
    "target": "cpu",
    "mode": "jit-calculate"
})
