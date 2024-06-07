"""
A module that contains functions to perform the forward and reverse Number-Theoretic Transform (NTT).
"""

from __future__ import annotations

import numpy as np

from ._fields import Field, FieldArray
from ._helper import export, verify_isinstance
from ._prime import is_prime
from .typing import ArrayLike


@export
def ntt(
    x: ArrayLike,
    size: int | None = None,
    modulus: int | None = None,
) -> FieldArray:
    r"""
    Computes the Number-Theoretic Transform (NTT) of $x$.

    Arguments:
        x: The input sequence of integers $x$.
        size: The size $N$ of the NTT transform, must be at least the length of $x$. The default is `None`
            which corresponds to `len(x)`. If `size` is larger than the length of $x$, $x$ is zero-padded.
        modulus: The prime modulus $p$ that defines the field $\mathrm{GF}(p)$. The prime modulus must
            satisfy $p > \textrm{max}(x)$ and $p = mN + 1$ (i.e., the size of the transform $N$ must
            divide $p - 1$). The default is `None` which corresponds to the smallest $p$ that satisfies
            the criteria. However, if $x$ is a $\mathrm{GF}(p)$ array, then `None` corresponds to
            $p$ from the specified field.

    Returns:
        The NTT $X$ of the input $x$, with length $N$. The output is a $\mathrm{GF}(p)$ array.
        It can be viewed as a normal NumPy array with `.view(np.ndarray)` or converted to a Python list with
        `.tolist()`.

    See Also:
        intt

    Notes:
        The Number-Theoretic Transform (NTT) is a specialized Discrete Fourier Transform (DFT) over a finite field
        $\mathrm{GF}(p)$ instead of over $\mathbb{C}$. The DFT uses the primitive $N$-th root of
        unity $\omega_N = e^{-i 2 \pi /N}$, but the NTT uses a primitive $N$-th root of unity in
        $\mathrm{GF}(p)$. These roots are such that $\omega_N^N = 1$ and $\omega_N^k \neq 1$
        for $0 < k < N$.

        In $\mathrm{GF}(p)$, where $p$ is prime, a primitive $N$-th root of unity exists if
        $N$ divides $p - 1$. If that is true, then $p = mN + 1$ for some integer $m$. This
        function finds $\omega_N$ by first finding a primitive $p - 1$-th root of unity
        $\omega_{p - 1}$ in $\mathrm{GF}(p)$ using :func:`~galois.primitive_root`. From there
        $\omega_N$ is found from $\omega_N = \omega_{p - 1}^m$.

        The $k$-th value of the $N$-point NTT $X = \mathrm{NTT}(x)$ is

        $$X_k = \sum_{j=0}^{N-1} x_j \omega_N^{jk} ,$$

        with all arithmetic performed in $\mathrm{GF}(p)$.

        A radix-2 Cooley-Tukey FFT algorithm is implemented, which achieves $O(N \mathrm{log}(N))$.

    References:
        - https://cgyurgyik.github.io/posts/2021/04/brief-introduction-to-ntt/
        - https://www.nayuki.io/page/number-theoretic-transform-integer-dft
        - https://www.geeksforgeeks.org/python-number-theoretic-transformation/

    Examples:
        The default modulus is the smallest $p$ such that $p > \textrm{max}(x)$ and $p = mN + 1$.
        With the input $x = [1, 2, 3, 4]$ and $N = 4$, the default modulus is $p = 5$.

        .. ipython:: python

            galois.ntt([1, 2, 3, 4])

        However, other moduli satisfy $p > \textrm{max}(x)$ and $p = mN + 1$. For instance, $p = 13$
        and $p = 17$. Notice the NTT outputs are different with different moduli. So it is important to perform
        forward and reverse NTTs with the same modulus.

        .. ipython:: python

            galois.ntt([1, 2, 3, 4], modulus=13)
            galois.ntt([1, 2, 3, 4], modulus=17)

        Instead of explicitly specifying the prime modulus, a $\mathrm{GF}(p)$ array may be explicitly passed in
        and the modulus is taken as $p$.

        .. ipython:: python

            GF = galois.GF(13)
            galois.ntt(GF([1, 2, 3, 4]))

        The `size` keyword argument allows convenient zero-padding of the input (to a power of two, for example).

        .. ipython:: python

            galois.ntt([1, 2, 3, 4, 5, 6], size=8)
            galois.ntt([1, 2, 3, 4, 5, 6, 0, 0])

        The :func:`numpy.fft.fft` function may also be used to compute the NTT over $\mathrm{GF}(p)$.

        .. ipython:: python

            GF = galois.GF(17)
            x = GF([1, 2, 3, 4, 5, 6])
            np.fft.fft(x, n=8)

    Group:
        transforms
    """
    verify_isinstance(x, (tuple, list, np.ndarray, FieldArray))
    if isinstance(x, FieldArray) and not type(x).is_prime_field:
        raise ValueError(f"If argument `x` is a FieldArray, it must be a prime field, not {type(x)}.")

    if modulus is None and isinstance(x, FieldArray):
        modulus = type(x).characteristic

    return _ntt(x, size=size, modulus=modulus, forward=True)


@export
def intt(
    X: ArrayLike,
    size: int | None = None,
    modulus: int | None = None,
    scaled: bool = True,
) -> FieldArray:
    r"""
    Computes the Inverse Number-Theoretic Transform (INTT) of $X$.

    Arguments:
        X: The input sequence of integers $X$.
        size: The size $N$ of the INTT transform, must be at least the length of $X$. The default is `None`
            which corresponds to `len(X)`. If `size` is larger than the length of $X$, $X$ is zero-padded.
        modulus: The prime modulus $p$ that defines the field $\mathrm{GF}(p)$. The prime modulus must
            satisfy $p > \textrm{max}(X)$ and $p = mN + 1$ (i.e., the size of the transform $N$
            must divide $p - 1$). The default is `None` which corresponds to the smallest $p$ that
            satisfies the criteria. However, if $x$ is a $\mathrm{GF}(p)$ array, then `None` corresponds
            to $p$ from the specified field.
        scaled: Indicates to scale the INTT output by $N$. The default is `True`. If `True`,
            $x = \mathrm{INTT}(\mathrm{NTT}(x))$. If `False`, $Nx = \mathrm{INTT}(\mathrm{NTT}(x))$.

    Returns:
        The INTT $x$ of the input $X$, with length $N$. The output is a $\mathrm{GF}(p)$ array.
        It can be viewed as a normal NumPy array with `.view(np.ndarray)` or converted to a Python list with
        `.tolist()`.

    See Also:
        ntt

    Notes:
        The Number-Theoretic Transform (NTT) is a specialized Discrete Fourier Transform (DFT) over a finite field
        $\mathrm{GF}(p)$ instead of over $\mathbb{C}$. The DFT uses the primitive $N$-th root of
        unity $\omega_N = e^{-i 2 \pi /N}$, but the NTT uses a primitive $N$-th root of unity in
        $\mathrm{GF}(p)$. These roots are such that $\omega_N^N = 1$ and $\omega_N^k \neq 1$ for
        $0 < k < N$.

        In $\mathrm{GF}(p)$, where $p$ is prime, a primitive $N$-th root of unity exists if
        $N$ divides $p - 1$. If that is true, then $p = mN + 1$ for some integer $m$. This
        function finds $\omega_N$ by first finding a primitive $p - 1$-th root of unity
        $\omega_{p - 1}$ in $\mathrm{GF}(p)$ using :func:`~galois.primitive_root`. From there
        $\omega_N$ is found from $\omega_N = \omega_{p - 1}^m$.

        The $j$-th value of the scaled $N$-point INTT $x = \mathrm{INTT}(X)$ is

        $$x_j = \frac{1}{N} \sum_{k=0}^{N-1} X_k \omega_N^{-kj} ,$$

        with all arithmetic performed in $\mathrm{GF}(p)$. The scaled INTT has the property that
        $x = \mathrm{INTT}(\mathrm{NTT}(x))$.

        A radix-2 Cooley-Tukey FFT algorithm is implemented, which achieves $O(N \mathrm{log}(N))$.

    References:
        - https://cgyurgyik.github.io/posts/2021/04/brief-introduction-to-ntt/
        - https://www.nayuki.io/page/number-theoretic-transform-integer-dft
        - https://www.geeksforgeeks.org/python-number-theoretic-transformation/

    Examples:
        The default modulus is the smallest $p$ such that $p > \textrm{max}(X)$ and $p = mN + 1$.
        With the input $X = [0, 4, 3, 2]$ and $N = 4$, the default modulus is $p = 5$.

        .. ipython:: python

            galois.intt([0, 4, 3, 2])

        However, other moduli satisfy $p > \textrm{max}(X)$ and $p = mN + 1$. For instance, $p = 13$
        and $p = 17$. Notice the INTT outputs are different with different moduli. So it is important to perform
        forward and reverse NTTs with the same modulus.

        .. ipython:: python

            galois.intt([0, 4, 3, 2], modulus=13)
            galois.intt([0, 4, 3, 2], modulus=17)

        Instead of explicitly specifying the prime modulus, a $\mathrm{GF}(p)$ array may be explicitly passed in
        and the modulus is taken as $p$.

        .. ipython:: python

            GF = galois.GF(13)
            X = GF([10, 8, 11, 1]); X
            x = galois.intt(X); x
            galois.ntt(x)

        The forward NTT and scaled INTT are the identity transform, i.e. $x = \mathrm{INTT}(\mathrm{NTT}(x))$.

        .. ipython:: python

            GF = galois.GF(13)
            x = GF([1, 2, 3, 4]); x
            galois.intt(galois.ntt(x))

        This is also true in the reverse order, i.e. $x = \mathrm{NTT}(\mathrm{INTT}(x))$.

        .. ipython:: python

            galois.ntt(galois.intt(x))

        The :func:`numpy.fft.ifft` function may also be used to compute the inverse NTT over $\mathrm{GF}(p)$.

        .. ipython:: python

            X = np.fft.fft(x); X
            np.fft.ifft(X)

    Group:
        transforms
    """
    verify_isinstance(X, (tuple, list, np.ndarray, FieldArray))
    if isinstance(X, FieldArray) and not type(X).is_prime_field:
        raise ValueError(f"If argument `X` is a FieldArray, it must be a prime field, not {type(X)}.")

    if modulus is None and isinstance(X, FieldArray):
        modulus = type(X).characteristic

    return _ntt(X, size=size, modulus=modulus, forward=False, scaled=scaled)


def _ntt(x, size=None, modulus=None, forward=True, scaled=True):
    verify_isinstance(size, int, optional=True)
    verify_isinstance(modulus, int, optional=True)
    verify_isinstance(forward, bool)
    verify_isinstance(scaled, bool)

    # The size N of the input/output sequence
    if size is None:
        size = len(x)

    # The prime modulus `p = m*N + 1` that defines the prime field GF(p)
    if modulus is None:
        m = int(np.ceil(np.max(x) / size))  # The smallest m such that modulus > max(x)
        while not is_prime(m * size + 1):
            m += 1
        modulus = m * size + 1
    m = (modulus - 1) // size

    if not size >= len(x):
        raise ValueError(f"Argument 'size' must be at least the length of the input which is {len(x)}, not {size}.")
    if not is_prime(modulus):
        raise ValueError(f"Argument 'modulus' must be prime, {modulus} is not.")
    if not (modulus - 1) % size == 0:
        raise ValueError("Argument 'modulus' must equal m * size + 1, where 'size' is the size of the NTT transform.")
    if not modulus > np.max(x):
        raise ValueError(
            f"Argument 'modulus' must be at least the max value of the input which is {np.max(x)}, not {modulus}."
        )

    field = Field(modulus)  # The prime field GF(p)

    x = field(x)
    if forward:
        norm = "forward" if scaled else "backward"
        y = np.fft.fft(x, n=size)
    else:
        norm = "backward" if scaled else "forward"
        y = np.fft.ifft(x, n=size, norm=norm)

    return y
