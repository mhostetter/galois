import numpy as np

from ..dtypes import DTYPES
from ..modular import euler_totient, is_cyclic, primitive_root, primitive_roots, totatives
from ..meta import Meta
from ..overrides import set_module

from .meta_ufunc_add import AdditiveGroupUfunc
from .meta_ufunc_mul import MultiplicativeGroupUfunc
from .meta_func import GroupFunc

__all__ = ["GroupMeta"]


@set_module("galois")
class GroupMeta(Meta):
    """
    Defines a metaclass for all :obj:`galois.GroupArray` classes.
    """
    # pylint: disable=no-value-for-parameter,comparison-with-callable,too-many-public-methods,abstract-method

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._modulus = kwargs.get("modulus", None)
        cls._operator = kwargs.get("operator", None)
        cls._generator = kwargs.get("generator", None)

        if cls.modulus is not None:
            cls._is_abelian = True
            cls._rank = 0
            cls._order_str = "n={}".format(cls.modulus)

            if cls.operator == "+":
                cls._is_cyclic = True
                cls._order = cls.modulus
                cls._generator = cls(1)
                cls._identity = cls(0)
            else:
                cls._is_cyclic = is_cyclic(cls.modulus)
                cls._order = euler_totient(cls.modulus)
                cls._generator = cls(primitive_root(cls.modulus)) if cls.is_cyclic else None  # pylint: disable=using-constant-test
                cls._identity = cls(1)

            cls.compile(kwargs["mode"], kwargs["target"])

    ###############################################################################
    # Array display methods
    ###############################################################################

    def _formatter(cls, array):  # pylint: disable=unused-argument
        formatter = {}
        formatter["object"] = cls._print_int
        return formatter

    def _print_int(cls, element):  # pylint: disable=no-self-use
        return "{:d}".format(int(element))

    ###############################################################################
    # Class attributes
    ###############################################################################

    @property
    def structure(cls):
        if cls.operator == "+":
            return "Finite Additive Group"
        else:
            return "Finite Multiplicative Group"

    @property
    def short_name(cls):
        return f"ℤn{cls.operator}"

    @property
    def name(cls):
        return f"(ℤ/{cls.modulus}ℤ){cls.operator}"

    @property
    def modulus(cls):
        """
        str: The modulus :math:`n` of the group :math:`(\\mathbb{Z}/n\\mathbb{Z}){^+}` or :math:`(\\mathbb{Z}/n\\mathbb{Z}){^\\times}`.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.modulus
            G = galois.Group(16, "*"); G.modulus
        """
        return cls._modulus

    @property
    def order(cls):
        """
        str: The order of the group, which equals the number of elements in the group.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.order
            G = galois.Group(16, "*"); G.order
        """
        return cls._order

    @property
    def generator(cls):
        """
        int: A generator of the group, if it exists. The group must be cyclic for a generator
        to exist. If a generator exists, the group can be represented as :math:`G = \\{g^0, g^1, \\dots, g^{\\phi(n)-1}\\}`.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.generator
            # This group doesn't have a generator and returns None
            G = galois.Group(16, "*"); G.generator
            G = galois.Group(17, "*"); G.generator
        """
        return cls._generator

    @property
    def generators(cls):
        """
        list: A list of all generators of the group. The group must be cyclic for a generator
        to exist. If a generator exists, the group can be represented as :math:`G = \\{g^0, g^1, \\dots, g^{\\phi(n)-1}\\}`

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.generators
            G = galois.Group(16, "*"); G.generators
            G = galois.Group(17, "*"); G.generators
        """
        if not hasattr(cls, "_generators"):
            if cls.operator == "+":
                cls._generators = cls.Elements()[1:]
            else:
                cls._generators = cls(primitive_roots(cls.modulus))
        return cls._generators

    # @property
    # def rank(cls):
    #     """
    #     int: The rank of the abelian group. *Rank* is defined as the maximal cardinality of a set of linearly independent
    #     elements of the group.
    #     """
    #     return cls._rank

    @property
    def set(cls):
        """
        set: The set of group elements.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.set
            G = galois.Group(16, "*"); G.set
            G = galois.Group(17, "*"); G.set
        """
        if not hasattr(cls, "_set"):
            if cls.operator == "+":
                cls._set = set(range(0, cls.modulus))
            else:
                cls._set = set(totatives(cls.modulus))
        return cls._set

    @property
    def operator(cls):
        """
        str: The group operator, either `"+"` or `"*"`.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.operator
            G = galois.Group(16, "*"); G.operator
        """
        return cls._operator

    @property
    def identity(cls):
        """
        int: The group identity element :math:`e`, such that :math:`a + e = a` for :math:`a, e \\in (\\mathbb{Z}/n\\mathbb{Z}){^+}`
        and :math:`a * e = a` for :math:`a, e \\in (\\mathbb{Z}/n\\mathbb{Z}){^\\times}`.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.identity
            G = galois.Group(16, "*"); G.identity
        """
        return cls._identity

    @property
    def is_cyclic(cls):
        """
        bool: Indicates if the group is cyclic. A group is *cyclic* if it can be generated by a generator
        element :math:`g` such that :math:`G = \\{g^0, g^1, \\dots, g^{\\phi(n)-1}\\}`.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+"); G.is_cyclic
            G = galois.Group(16, "*"); G.is_cyclic
            G = galois.Group(17, "*"); G.is_cyclic
        """
        return cls._is_cyclic

    @property
    def is_abelian(cls):
        """
        bool: Indicates if the group is abelian. A group is *abelian* if the order of elements
        in the group operation does not matter.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+")
            G.is_abelian
            a, b = G.Random(), G.Random()
            a + b
            b + a

        .. ipython:: python

            G = galois.Group(16, "*")
            G.is_abelian
            a, b = G.Random(), G.Random()
            a * b
            b * a
        """
        return cls._is_abelian

    @property
    def dtypes(cls):
        max_dtype = DTYPES[-1]
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.modulus - 1 and np.iinfo(max_dtype).max >= (cls.modulus - 1)**2]
        if len(d) == 0:
            d = [np.object_]
        return d

    @property
    def ufunc_modes(cls):
        if cls.dtypes == [np.object_]:
            return ["python-calculate"]
        else:
            return ["jit-calculate"]

    @property
    def default_ufunc_mode(cls):
        if cls.dtypes == [np.object_]:
            return "python-calculate"
        else:
            return "jit-calculate"

    @property
    def properties(cls):
        string = f"{cls.name}:"
        string += f"\n  structure: {cls.structure}"
        string += f"\n  modulus: {cls.modulus}"
        string += f"\n  order: {cls.order}"
        string += f"\n  generator: {cls.generator}"
        string += f"\n  is_cyclic: {cls.is_cyclic}"
        string += f"\n  is_abelian: {cls.is_abelian}"
        return string


class AdditiveGroupMeta(GroupMeta, AdditiveGroupUfunc, GroupFunc):
    """
    Defines a metaclass for all :obj:`galois.GroupArray` classes with the addition operator.
    """


class MultiplicativeGroupMeta(GroupMeta, MultiplicativeGroupUfunc, GroupFunc):
    """
    Defines a metaclass for all :obj:`galois.GroupArray` classes with the multiplication operator.
    """
