"""
A module that contains functions to perform the forward and reverse Number-Theoretic Transform (NTT).
"""
from typing import Tuple, List, Optional, Union

import numpy as np

from ._fields import Field, FieldArray
from ._modular import primitive_root
from ._overrides import set_module
from ._prime import is_prime

__all__ = ["ntt", "intt"]


@set_module("galois")
def ntt(
    x: Union[Tuple[int], List[int], np.ndarray, FieldArray],
    size: Optional[int] = None,
    modulus: Optional[int] = None
) -> FieldArray:
    r"""
    Computes the Number-Theoretic Transform (NTT) of :math:`x`.

    Parameters
    ----------
    x : tuple, list, numpy.ndarray, galois.FieldArray
        The input sequence of integers :math:`x`.
    size : int, optional
        The size :math:`N` of the NTT transform, must be at least the length of :math:`x`. The default is `None` which corresponds to `len(x)`.
        If `size` is larger than the length of :math:`x`, :math:`x` is zero-padded.
    modulus : int, optional
        The prime modulus :math:`p` that defines the field :math:`\mathrm{GF}(p)`. The prime modulus must satisfy :math:`p > \textrm{max}(x)`
        and :math:`p = mN + 1` (i.e., the size of the transform :math:`N` must divide :math:`p - 1`). The default is `None` which corresponds
        to the smallest :math:`p` that satisfies the criteria. However, if :math:`x` is a :math:`\mathrm{GF}(p)` array, then `None` corresponds
        to :math:`p` from the specified field.

    Returns
    -------
    galois.FieldArray
        The NTT :math:`X` of the input :math:`x`, with length :math:`N`. The output is a :math:`\mathrm{GF}(p)` array. It can be viewed
        as a normal NumPy array with `.view(np.ndarray)` or converted to a Python list with `.tolist()`.

    Notes
    -----
    The Number-Theoretic Transform (NTT) is a specialized Discrete Fourier Transform (DFT) over a finite field :math:`\mathrm{GF}(p)`
    instead of over :math:`\mathbb{C}`. The DFT uses the primitive :math:`N`-th root of unity :math:`\omega_N = e^{-i 2 \pi /N}`,
    but the NTT uses a primitive :math:`N`-th root of unity in :math:`\mathrm{GF}(p)`. These roots are such that :math:`\omega_N^N = 1` and
    :math:`\omega_N^k \neq 1` for :math:`0 < k < N`.

    In :math:`\mathrm{GF}(p)`, where :math:`p` is prime, a primitive :math:`N`-th root of unity exists if :math:`N` divides :math:`p - 1`. If that is true,
    then :math:`p = mN + 1` for some integer :math:`m`. This function finds :math:`\omega_N` by first finding a primitive :math:`p - 1`-th root of
    unity :math:`\omega_{p - 1}` in :math:`\mathrm{GF}(p)` using :func:`galois.primitive_root`. From there :math:`\omega_N` is found
    from :math:`\omega_N = \omega_{p - 1}^m`.

    The :math:`k`-th value of the :math:`N`-point NTT :math:`X = \mathrm{NTT}(x)` is

    .. math::

        X_k = \sum_{j=0}^{N-1} x_j \omega_N^{jk} ,

    with all arithmetic performed in :math:`\mathrm{GF}(p)`.

    Currently, the algorithm implemented is :math:`O(N^2)`. A future improvement will be to add a radix-2 Cooley-Tukey implementation,
    which will have :math:`O(N \mathrm{log}(N))`.

    References
    ----------
    * https://cgyurgyik.github.io/posts/2021/04/brief-introduction-to-ntt/
    * https://www.nayuki.io/page/number-theoretic-transform-integer-dft
    * https://www.geeksforgeeks.org/python-number-theoretic-transformation/

    Examples
    --------
    The default modulus is the smallest :math:`p` such that :math:`p > \textrm{max}(x)` and :math:`p = mN + 1`. With the input
    :math:`x = [1, 2, 3, 4]` and :math:`N = 4`, the default modulus is :math:`p = 5`.

    .. ipython:: python

        galois.ntt([1, 2, 3, 4])

    However, other moduli satisfy :math:`p > \textrm{max}(x)` and :math:`p = mN + 1`. For instance, :math:`p = 13` and :math:`p = 17`.
    Notice the NTT outputs are different with different moduli. So it is important to perform forward and reverse NTTs with the same modulus.

    .. ipython:: python

        galois.ntt([1, 2, 3, 4], modulus=13)
        galois.ntt([1, 2, 3, 4], modulus=17)

    Instead of explicitly specifying the prime modulus, a :math:`\mathrm{GF}(p)` array may be explicitly passed in and the modulus
    is taken as :math:`p`.

    .. ipython:: python

        GF = galois.GF(13)
        galois.ntt(GF([1, 2, 3, 4]))

    The `size` keyword argument allows convenient zero-padding of the input (to a power of two, for example).

    .. ipython:: python

        galois.ntt([1, 2, 3, 4, 5, 6], size=8)
        galois.ntt([1, 2, 3, 4, 5, 6, 0, 0])
    """
    if not isinstance(x, (tuple, list, np.ndarray, FieldArray)):
        raise TypeError(f"Argument `x` must be array-like, not {type(x)}.")
    if isinstance(x, FieldArray) and not type(x).is_prime_field:
        raise ValueError(f"If argument `x` is a Galois field array, it must be a prime field, not {type(x)}.")

    if modulus is None and isinstance(x, FieldArray):
        modulus = type(x).characteristic

    return _ntt(x, size=size, modulus=modulus, forward=True)


@set_module("galois")
def intt(
    X: Union[Tuple[int], List[int], np.ndarray, FieldArray],
    size: Optional[int] = None,
    modulus: Optional[int] = None,
    scaled: bool = True
) -> FieldArray:
    r"""
    Computes the Inverse Number-Theoretic Transform (INTT) of :math:`X`.

    Parameters
    ----------
    X : tuple, list, numpy.ndarray, galois.FieldArray
        The input sequence of integers :math:`X`.
    size : int, optional
        The size :math:`N` of the INTT transform, must be at least the length of :math:`X`. The default is `None` which corresponds to `len(X)`.
        If `size` is larger than the length of :math:`X`, :math:`X` is zero-padded.
    modulus : int, optional
        The prime modulus :math:`p` that defines the field :math:`\mathrm{GF}(p)`. The prime modulus must satisfy :math:`p > \textrm{max}(X)`
        and :math:`p = mN + 1` (i.e., the size of the transform :math:`N` must divide :math:`p - 1`).The default is `None` which corresponds
        to the smallest :math:`p` that satisfies the criteria. However, if :math:`x` is a :math:`\mathrm{GF}(p)` array, then `None` corresponds
        to :math:`p` from the specified field.
    scaled : bool, optional
        Indicates to scale the INTT output by :math:`N`. The default is `True`. If true, :math:`x = \mathrm{INTT}(\mathrm{NTT}(x))`. If false,
        :math:`Nx = \mathrm{INTT}(\mathrm{NTT}(x))`.

    Returns
    -------
    galois.FieldArray
        The INTT :math:`x` of the input :math:`X`, with length :math:`N`. The output is a :math:`\mathrm{GF}(p)` array. It can be viewed
        as a normal NumPy array with `.view(np.ndarray)` or converted to a Python list with `.tolist()`.

    Notes
    -----
    The Number-Theoretic Transform (NTT) is a specialized Discrete Fourier Transform (DFT) over a finite field :math:`\mathrm{GF}(p)`
    instead of over :math:`\mathbb{C}`. The DFT uses the primitive :math:`N`-th root of unity :math:`\omega_N = e^{-i 2 \pi /N}`,
    but the NTT uses a primitive :math:`N`-th root of unity in :math:`\mathrm{GF}(p)`. These roots are such that :math:`\omega_N^N = 1` and
    :math:`\omega_N^k \neq 1` for :math:`0 < k < N`.

    In :math:`\mathrm{GF}(p)`, where :math:`p` is prime, a primitive :math:`N`-th root of unity exists if :math:`N` divides :math:`p - 1`. If that is true,
    then :math:`p = mN + 1` for some integer :math:`m`. This function finds :math:`\omega_N` by first finding a primitive :math:`p - 1`-th root of
    unity :math:`\omega_{p - 1}` in :math:`\mathrm{GF}(p)` using :func:`galois.primitive_root`. From there :math:`\omega_N` is found
    from :math:`\omega_N = \omega_{p - 1}^m`.

    The :math:`j`-th value of the scaled :math:`N`-point INTT :math:`x = \mathrm{INTT}(X)` is

    .. math::

        x_j = \frac{1}{N} \sum_{k=0}^{N-1} X_k \omega_N^{-kj} ,

    with all arithmetic performed in :math:`\mathrm{GF}(p)`. The scaled INTT has the property that :math:`x = \mathrm{INTT}(\mathrm{NTT}(x))`.

    Currently, the algorithm implemented is :math:`O(N^2)`. A future improvement will be to add a radix-2 Cooley-Tukey implementation,
    which will have :math:`O(N \mathrm{log}(N))`.

    References
    ----------
    * https://cgyurgyik.github.io/posts/2021/04/brief-introduction-to-ntt/
    * https://www.nayuki.io/page/number-theoretic-transform-integer-dft
    * https://www.geeksforgeeks.org/python-number-theoretic-transformation/

    Examples
    --------
    The default modulus is the smallest :math:`p` such that :math:`p > \textrm{max}(X)` and :math:`p = mN + 1`. With the input
    :math:`X = [0, 4, 3, 2]` and :math:`N = 4`, the default modulus is :math:`p = 5`.

    .. ipython:: python

        galois.intt([0, 4, 3, 2])

    However, other moduli satisfy :math:`p > \textrm{max}(X)` and :math:`p = mN + 1`. For instance, :math:`p = 13` and :math:`p = 17`.
    Notice the INTT outputs are different with different moduli. So it is important to perform forward and reverse NTTs with the same modulus.

    .. ipython:: python

        galois.intt([0, 4, 3, 2], modulus=13)
        galois.intt([0, 4, 3, 2], modulus=17)

    Instead of explicitly specifying the prime modulus, a :math:`\mathrm{GF}(p)` array may be explicitly passed in and the modulus
    is taken as :math:`p`.

    .. ipython:: python

        GF = galois.GF(13)
        X = GF([10, 8, 11, 1]); X
        x = galois.intt(X); x
        galois.ntt(x)

    The forward NTT and scaled INTT are the identity transform, i.e. :math:`x = \mathrm{INTT}(\mathrm{NTT}(x))`.

    .. ipython:: python

        GF = galois.GF(13)
        x = GF([1, 2, 3, 4]); x
        galois.intt(galois.ntt(x))

    This is also true in the reverse order, i.e. :math:`x = \mathrm{NTT}(\mathrm{INTT}(x))`.

    .. ipython:: python

        galois.ntt(galois.intt(x))
    """
    if not isinstance(X, (tuple, list, np.ndarray, FieldArray)):
        raise TypeError(f"Argument `X` must be array-like, not {type(X)}.")
    if isinstance(X, FieldArray) and not type(X).is_prime_field:
        raise ValueError(f"If argument `X` is a Galois field array, it must be a prime field, not {type(X)}.")

    if modulus is None and isinstance(X, FieldArray):
        modulus = type(X).characteristic

    return _ntt(X, size=size, modulus=modulus, forward=False, scaled=scaled)


def _ntt(x, size=None, modulus=None, forward=True, scaled=True):
    if not isinstance(size, (type(None), int, np.integer)):
        raise TypeError(f"Argument `size` must be an integer, not {type(size)}.")
    if not isinstance(modulus, (type(None), int, np.integer)):
        raise TypeError(f"Argument `modulus` must be an integer, not {type(modulus)}.")
    if not isinstance(forward, bool):
        raise TypeError(f"Argument `forward` must be a bool, not {type(forward)}.")
    if not isinstance(scaled, bool):
        raise TypeError(f"Argument `scaled` must be a bool, not {type(scaled)}.")

    # The size N of the input/output sequence
    if size is None:
        size = len(x)

    # The prime modulus `p = m*N + 1` that defines the prime field GF(p)
    if modulus is None:
        m = int(np.ceil(np.max(x) / size))  # The smallest m such that modulus > max(x)
        while not is_prime(m*size + 1):
            m += 1
        modulus = m*size + 1
    m = (modulus - 1) // size

    if not size >= len(x):
        raise ValueError(f"Argument `size` must be at least the length of the input which is {len(x)}, not {size}.")
    if not is_prime(modulus):
        raise ValueError(f"Argument `modulus` must be prime, {modulus} is not.")
    if not (modulus - 1) % size == 0:
        raise ValueError("Argument `modulus` must satisfy `modulus = m*size + 1` where `size` is the size of the NTT transform.")
    if not modulus > np.max(x):
        raise ValueError(f"Argument `modulus` must be at least the max value of the input which is {np.max(x)}, not {modulus}.")

    field = Field(modulus)  # The prime field GF(p)
    xx = field.Zeros(size)
    xx[0:len(x)] = x  # Potentially zero-pad the input to length `size`

    g = primitive_root(modulus)  # A generator of the multiplicative group of GF(p)
    omega = field(g)**m if forward else field(g)**-m  # A primitive N-th root of unity in GF(p)
    v = field.Vandermonde(omega, size, size)

    # Transform x into y with the NTT by matrix multiplying x with the Vandermonde matrix of primitive N-th root of unity `omega`
    y = v @ xx

    # Scale the inverse NTT such that x = INTT(NTT(x))
    if not forward and scaled:
        y /= field(size)

    return y
