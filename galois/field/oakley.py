from ..overrides import set_module

from .factory import GF
from .poly import Poly

__all__ = ["Oakley1", "Oakley2", "Oakley3", "Oakley4"]


@set_module("galois")
def Oakley1():
    """
    Returns the Galois field for the first Oakley group from RFC 2409.

    References
    ----------
    * https://datatracker.ietf.org/doc/html/rfc2409#section-6.1

    Examples
    --------
    .. ipython:: python

        GF = galois.Oakley1()
        print(GF.properties)
    """
    prime = 0xFFFFFFFF_FFFFFFFF_C90FDAA2_2168C234_C4C6628B_80DC1CD1_29024E08_8A67CC74_020BBEA6_3B139B22_514A0879_8E3404DD_EF9519B3_CD3A431B_302B0A6D_F25F1437_4FE1356D_6D51C245_E485B576_625E7EC6_F44C42E9_A63A3620_FFFFFFFF_FFFFFFFF
    generator = 2
    return GF(prime, primitive_element=generator, verify=False)


@set_module("galois")
def Oakley2():
    """
    Returns the Galois field for the second Oakley group from RFC 2409.

    References
    ----------
    * https://datatracker.ietf.org/doc/html/rfc2409#section-6.2

    Examples
    --------
    .. ipython:: python

        GF = galois.Oakley2()
        print(GF.properties)
    """
    prime = 0xFFFFFFFF_FFFFFFFF_C90FDAA2_2168C234_C4C6628B_80DC1CD1_29024E08_8A67CC74_020BBEA6_3B139B22_514A0879_8E3404DD_EF9519B3_CD3A431B_302B0A6D_F25F1437_4FE1356D_6D51C245_E485B576_625E7EC6_F44C42E9_A637ED6B_0BFF5CB6_F406B7ED_EE386BFB_5A899FA5_AE9F2411_7C4B1FE6_49286651_ECE65381_FFFFFFFF_FFFFFFFF
    generator = 2
    return GF(prime, primitive_element=generator, verify=False)


@set_module("galois")
def Oakley3():
    """
    Returns the Galois field for the third Oakley group from RFC 2409.

    References
    ----------
    * https://datatracker.ietf.org/doc/html/rfc2409#section-6.3

    Examples
    --------
    .. ipython:: python

        GF = galois.Oakley3()
        print(GF.properties)
    """
    degree = 155
    irreducible_poly = Poly.Integer(0x0800000000000000000000004000000000000001)
    primitive_element = Poly.Integer(0x7b)
    return GF(2**degree, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=False)


@set_module("galois")
def Oakley4():
    """
    Returns the Galois field for the fourth Oakley group from RFC 2409.

    References
    ----------
    * https://datatracker.ietf.org/doc/html/rfc2409#section-6.4

    Examples
    --------
    .. ipython:: python

        GF = galois.Oakley4()
        print(GF.properties)
    """
    degree = 185
    irreducible_poly = Poly.Integer(0x020000000000000000000000000000200000000000000001)
    primitive_element = Poly.Integer(0x18)
    return GF(2**degree, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=False)
