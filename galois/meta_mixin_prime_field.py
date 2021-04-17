from .meta_mixin_target import TargetMixin


class PrimeFieldMixin(TargetMixin):
    """
    A mixin class that constructs lookup tables for prime fields: GF2 and GFp.
    """
    # pylint: disable=abstract-method,no-value-for-parameter

    def _fill_in_lookup_tables(cls):
        """
        Constructs lookup tables for prime fields.
        """
        order = cls.order
        primitive_element = int(cls._primitive_element)

        element = 1
        cls._EXP[0] = element
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, order):
            # Increment by multiplying by the primitive element, which is a multiplicative generator of the field
            element *= primitive_element
            if element >= order:
                element %= order
            cls._EXP[i] = element

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, order):
            val = (1 + cls._EXP[i]) % order  # Addition in GF(p)
            cls._ZECH_LOG[i] = cls._LOG[val]
