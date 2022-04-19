from .._domains._array import Meta
from .._polys import Poly
from .._polys._conversions import integer_to_poly, poly_to_str


class FieldMeta(Meta):
    """
    A metaclass that provides class properties for `FieldArray` subclasses.
    """

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly_int: int = kwargs.get("irreducible_poly_int", 0)
        cls._is_primitive_poly: bool = kwargs.get("is_primitive_poly", None)
        cls._primitive_element: int = kwargs.get("primitive_element", 0)

        if cls._degree == 1:
            cls._prime_subfield = cls
            cls._name = f"GF({cls._characteristic})"
            cls._order_str = f"order={cls._order}"
        else:
            cls._prime_subfield = kwargs["prime_subfield"]  # Must be provided
            cls._name = f"GF({cls._characteristic}^{cls._degree})"
            cls._order_str = f"order={cls._characteristic}^{cls._degree}"

        # Construct the irreducible polynomial from its integer representation
        cls._irreducible_poly = Poly.Int(cls._irreducible_poly_int, field=cls._prime_subfield)

        if "compile" in kwargs:
            cls.compile(kwargs["compile"])

    def __str__(cls) -> str:
        if cls._prime_subfield is None:
            return repr(cls)

        with cls._prime_subfield.display("int"):
            irreducible_poly_str = str(cls._irreducible_poly)

        string = "Galois Field:"
        string += f"\n  name: {cls._name}"
        string += f"\n  characteristic: {cls._characteristic}"
        string += f"\n  degree: {cls._degree}"
        string += f"\n  order: {cls._order}"
        string += f"\n  irreducible_poly: {irreducible_poly_str}"
        string += f"\n  is_primitive_poly: {cls._is_primitive_poly}"
        string += f"\n  primitive_element: {poly_to_str(integer_to_poly(int(cls._primitive_element), cls._characteristic))}"

        return string
