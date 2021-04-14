import numpy as np

from .poly import Poly
from .meta_mixin_target import TargetMixin


class ExtensionFieldMixin(TargetMixin):
    """
    A mixin class that constructs lookup tables for extension fields: GF2m and GFpm.
    """
    # pylint: disable=abstract-method,no-value-for-parameter

    def _build_lookup_tables(cls):
        order = cls.order
        ground_field = cls._ground_field
        prim_poly = cls.irreducible_poly
        one = Poly.One(ground_field)
        prim_element = cls._primitive_element

        dtype = np.int64
        if order > np.iinfo(dtype).max:
            raise ValueError(f"Cannot build lookup tables for GF(p) class with order {order} since the elements cannot be represented with dtype {dtype}")

        EXP = np.zeros(2*order, dtype=dtype)
        LOG = np.zeros(order, dtype=dtype)
        ZECH_LOG = np.zeros(order, dtype=dtype)

        element = one.copy()
        EXP[0] = element.integer
        LOG[0] = 0  # Technically -Inf
        for i in range(1, order):
            # Increment by multiplying by the primitive element, which is a "multiplicative generator" of the group
            element *= prim_element
            if element.degree >= cls.degree:
                element %= prim_poly
            EXP[i] = element.integer

            # Assign to the log lookup but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order-1]``
            if i < order - 1:
                LOG[EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, order):
            val = (one + Poly.Integer(EXP[i], ground_field)).integer  # Addition in GF(p^m)
            ZECH_LOG[i] = LOG[val]

        assert EXP[order - 1] == 1, f"Primitive polynomial {prim_poly} does not have multiplicative order {order-1} and therefore isn't a multiplicative generator for GF({order})."
        assert len(set(EXP[0:order - 1])) == order - 1, "The anti-log lookup table is not unique."
        assert len(set(LOG[1:order])) == order - 1, "The log lookup table is not unique."

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        EXP[order:2*order] = EXP[1:1 + order]

        return EXP, LOG, ZECH_LOG
