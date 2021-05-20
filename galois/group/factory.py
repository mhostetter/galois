import types

import numpy as np

from ..array import GroupArrayBase
from ..overrides import set_module

from .array import GroupArray
from .meta import AdditiveGroupMeta, MultiplicativeGroupMeta

__all__ = ["Group"]

# pylint: disable=protected-access


@set_module("galois")
def Group(modulus, operator):
    """
    Factory function to construct a finite group array class of type :math:`(\\mathbb{Z}/n\\mathbb{Z}){^+}` or :math:`(\\mathbb{Z}/n\\mathbb{Z}){^\\times}`.

    The created class will be a subclass of :obj:`galois.GroupArray` with metaclass :obj:`galois.GroupMeta`.
    The :obj:`galois.GroupArray` inheritance provides the :obj:`numpy.ndarray` functionality. The :obj:`galois.GroupMeta` metaclass
    provides a variety of class attributes and methods relating to the finite group.

    Parameters
    ----------
    modulus : int
        The modulus :math:`n` of the group.
    operator : str
        The group operation, either `"+"` or `"*"`.

    Returns
    -------
    galois.GroupMeta
        A new finite group array class that is a subclass of :obj:`galois.GroupArray` with :obj:`galois.GroupMeta` metaclass.

    Examples
    --------
    Construct a finite group array class for the additive group :math:`(\\mathbb{Z}/16\\mathbb{Z}){^+}`

    .. ipython:: python

        G = galois.Group(16, "+")
        print(G.properties)
        G.Elements()
        a = G.Random(5); a
        b = G.Random(5); b
        a + b

    Construct a finite group array class for the multiplicative group :math:`(\\mathbb{Z}/16\\mathbb{Z}){^\\times}`

    .. ipython:: python

        G = galois.Group(16, "*")
        # Notice this group is not cyclic
        print(G.properties)
        G.Elements()
        a = G.Random(5); a
        b = G.Random(5); b
        a * b
    """
    if not isinstance(modulus, (int, np.integer)):
        raise TypeError(f"Argument `modulus` must be an integer, not {type(modulus)}.")
    if not operator in ["+", "*"]:
        raise ValueError(f"Argument `operator` must be either '+' or '*', not {operator}.")
    mode = "auto"
    target = "cpu"

    key = (modulus, operator)
    if key in Group._classes:
        cls = Group._classes[key]
        cls.compile(mode, target)
        return cls

    if operator == "+":
        name = f"Z_{modulus}"
        cls = types.new_class(name, bases=(GroupArray, GroupArrayBase), kwds={
            "metaclass": AdditiveGroupMeta,
            "operator": "+",
            "modulus": modulus,
            "mode": mode,
            "target": target
        })
    else:
        name = f"Z_{modulus}"
        cls = types.new_class(name, bases=(GroupArray, GroupArrayBase), kwds={
            "metaclass": MultiplicativeGroupMeta,
            "operator": "*",
            "modulus": modulus,
            "mode": mode,
            "target": target
        })

    # Add class to dictionary of flyweights
    Group._classes[key] = cls

    return cls

Group._classes = {}
