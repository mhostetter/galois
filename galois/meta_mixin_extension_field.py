from .poly import Poly
from .meta_mixin_target import TargetMixin


class ExtensionFieldMixin(TargetMixin):
    """
    A mixin class that constructs lookup tables for extension fields: GF2m and GFpm.
    """
    # pylint: disable=abstract-method,no-value-for-parameter

    def _fill_in_lookup_tables(cls):
        order = cls.order
        ground_field = cls._ground_field
        prim_poly = cls.irreducible_poly
        one = Poly.One(ground_field)
        primitive_element = cls._primitive_element

        element = one.copy()
        cls._EXP[0] = element.integer
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, order):
            # Increment by multiplying by the primitive element, which is a multiplicative generator of the field
            element *= primitive_element
            if element.degree >= cls.degree:
                element %= prim_poly
            cls._EXP[i] = element.integer

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, order):
            val = (one + Poly.Integer(cls._EXP[i], ground_field)).integer  # Addition in GF(p^m)
            cls._ZECH_LOG[i] = cls._LOG[val]
