"""
A module to get or set package-wide options.
"""

import contextlib
from typing import Any, Dict, Generator

from typing_extensions import Literal

from ._helper import export

# The default print options for the package
PRINTOPTIONS = {}


@export
def set_printoptions(coeffs: Literal["desc", "asc"] = "desc"):
    """
    Modifies the print options for the package. This function is the :obj:`galois` equivalent of
    :func:`numpy.set_printoptions`.

    Arguments:
        coeffs: The order in which to print the coefficients, either in descending degrees (default) or ascending
            degrees.

    See Also:
        get_printoptions, printoptions

    Examples:
        By default, polynomials are displayed with descending degrees.

        .. ipython:: python

            GF = galois.GF(3**5, repr="poly")
            a = GF([109, 83]); a
            f = galois.Poly([3, 0, 5, 2], field=galois.GF(7)); f

        Modify the print options to display polynomials with ascending degrees.

        .. ipython:: python

            galois.set_printoptions(coeffs="asc")
            a
            f
            @suppress
            GF.repr()
            @suppress
            galois.set_printoptions()

    Group:
        config
    """
    if not coeffs in ["desc", "asc"]:
        raise ValueError(f"Argument 'coeffs' must be in ['desc', 'asc'], not {coeffs}.")

    PRINTOPTIONS["coeffs"] = coeffs


# Update the global print options with the default values
set_printoptions()


@export
def get_printoptions() -> Dict[str, Any]:
    """
    Returns the current print options for the package. This function is the :obj:`galois` equivalent of
    :func:`numpy.get_printoptions`.

    Returns:
        A dictionary of current print options.

    See Also:
        set_printoptions, printoptions

    Examples:
        .. ipython:: python

            galois.get_printoptions()

        .. ipython:: python

            galois.set_printoptions(coeffs="asc")
            galois.get_printoptions()
            @suppress
            galois.set_printoptions()

    Group:
        config
    """
    return PRINTOPTIONS.copy()


@export
@contextlib.contextmanager
def printoptions(**kwargs) -> Generator[None, None, None]:
    """
    A context manager to temporarily modify the print options for the package. This function is the :obj:`galois`
    equivalent of :func:`numpy.printoptions`.

    See :func:`~galois.set_printoptions` for the full list of available options.

    Returns:
        A context manager for use in a `with` statement. The print options are only modified inside the `with` block.

    See Also:
        set_printoptions, get_printoptions

    Examples:
        By default, polynomials are displayed with descending degrees.

        .. ipython:: python

            GF = galois.GF(3**5, repr="poly")
            a = GF([109, 83])
            f = galois.Poly([3, 0, 5, 2], field=galois.GF(7))

        Modify the print options only inside the context manager.

        .. ipython:: python

            print(a); print(f)
            with galois.printoptions(coeffs="asc"):
                print(a); print(f)
            print(a); print(f)
            @suppress
            GF.repr()

    Group:
        config
    """
    options = get_printoptions()
    set_printoptions(**kwargs)
    yield
    set_printoptions(**options)
