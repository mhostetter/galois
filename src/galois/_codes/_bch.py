"""
A module containing arbitrary Bose-Chaudhuri-Hocquenghem (BCH) codes over GF(2).
"""
from __future__ import annotations

import math
from typing import Tuple, List, Optional, Union, Type, overload
from typing_extensions import Literal

import numba
from numba import int64
import numpy as np

from .._domains._function import Function
from .._fields import Field, FieldArray, GF2
from .._helper import export, verify_isinstance
from .._lfsr import berlekamp_massey_jit
from .._polys import Poly, matlab_primitive_poly
from .._polys._dense import roots_jit, divmod_jit
from .._prime import factors
from ..typing import ArrayLike, PolyLike

from ._cyclic import poly_to_generator_matrix, roots_to_parity_check_matrix


def _check_and_compute_field(
    n: int,
    k: int,
    c: int,
    primitive_poly: Optional[PolyLike] = None,
    primitive_element: Optional[PolyLike] = None
) -> Type[FieldArray]:
    verify_isinstance(n, int)
    verify_isinstance(k, int)
    verify_isinstance(c, int)

    p, m = factors(n + 1)
    if not (len(p) == 1 and p[0] == 2):
        raise ValueError(f"Argument `n` must have value `2^m - 1` for some positive m, not {n}.")
    if not c >= 1:
        raise ValueError(f"Argument `c` must be at least 1, not {c}.")
    p, m = p[0], m[0]

    if primitive_poly is None:
        primitive_poly = matlab_primitive_poly(2, m)
    else:
        primitive_poly = Poly._PolyLike(primitive_poly)
    if primitive_element is not None:
        primitive_element = Poly._PolyLike(primitive_element)

    GF = Field(2**m, irreducible_poly=primitive_poly, primitive_element=primitive_element)

    return GF


@export
def bch_valid_codes(n: int, t_min: int = 1) -> List[Tuple[int, int, int]]:
    r"""
    Returns a list of :math:`(n, k, t)` tuples of valid primitive binary BCH codes.

    A BCH code with parameters :math:`(n, k, t)` is represented as a :math:`[n, k, d]_2` linear block
    code with :math:`d = 2t + 1`.

    Parameters
    ----------
    n
        The codeword size :math:`n`, must be :math:`n = 2^m - 1`.
    t_min
        The minimum error-correcting capability. The default is 1.

    Returns
    -------
    :
        A list of :math:`(n, k, t)` tuples of valid primitive BCH codes.

    See Also
    --------
    BCH

    References
    ----------
    * https://link.springer.com/content/pdf/bbm%3A978-1-4899-2174-1%2F1.pdf

    Examples
    --------
    .. ipython:: python

        galois.bch_valid_codes(31)
        galois.bch_valid_codes(31, t_min=3)

    :group: fec
    """
    verify_isinstance(n, int)
    verify_isinstance(t_min, int)
    if not t_min >= 1:
        raise ValueError(f"Argument `t_min` must be at least 1, not {t_min}.")

    GF = _check_and_compute_field(n, 0, 1, None, None)  # NOTE: k isn't needed for generating the field
    alpha = GF.primitive_element

    codes = []
    t = t_min
    while True:
        c = 1
        roots = alpha**(c + np.arange(0, 2*t))
        powers = GF.characteristic**np.arange(0, GF.degree, dtype=GF.dtypes[-1])
        conjugates = np.unique(np.power.outer(roots, powers))
        g_degree = len(conjugates)
        k = n - g_degree

        if k < 1:
            # There are no more valid codes
            break

        if len(codes) > 0 and codes[-1][1] == k:
            # If this code has the same size but more correcting power, replace it
            codes[-1] = (n, k, t)
        else:
            codes.append((n, k, t))

        t += 1

    return codes


@export
class BCH:
    r"""
    A primitive, narrow-sense binary :math:`\textrm{BCH}(n, k)` code.

    A :math:`\textrm{BCH}(n, k)` code is a :math:`[n, k, d]_2` linear block code with codeword size :math:`n`, message
    size :math:`k`, minimum distance :math:`d`, and symbols taken from an alphabet of size 2.

    To create the shortened :math:`\textrm{BCH}(n-s, k-s)` code, construct the full-sized :math:`\textrm{BCH}(n, k)` code
    and then pass :math:`k-s` bits into :func:`encode` and :math:`n-s` bits into :func:`decode()`. Shortened codes are only
    applicable for systematic codes.

    Examples
    --------
    Construct the BCH code.

    .. ipython:: python

        galois.bch_valid_codes(15)
        bch = galois.BCH(15, 7); bch

    Encode a message.

    .. ipython:: python

        m = galois.GF2.Random(bch.k); m
        c = bch.encode(m); c

    Corrupt the codeword and decode the message.

    .. ipython:: python

        # Corrupt the first bit in the codeword
        c[0] ^= 1
        dec_m = bch.decode(c); dec_m
        np.array_equal(dec_m, m)

    .. ipython:: python

        # Instruct the decoder to return the number of corrected bit errors
        dec_m, N = bch.decode(c, errors=True); dec_m, N
        np.array_equal(dec_m, m)

    :group: fec
    """
    # pylint: disable=no-member

    def __init__(
        self,
        n: int,
        k: int,
        primitive_poly: Optional[PolyLike] = None,
        primitive_element: Optional[PolyLike] = None,
        systematic: bool = True
    ):
        r"""
        Constructs a primitive, narrow-sense binary :math:`\textrm{BCH}(n, k)` code.

        Parameters
        ----------
        n
            The codeword size :math:`n`, must be :math:`n = 2^m - 1`.
        k
            The message size :math:`k`.
        primitive_poly
            Optionally specify the primitive polynomial that defines the extension field :math:`\mathrm{GF}(2^m)`. The default is
            `None` which uses Matlab's default, see :func:`~galois.matlab_primitive_poly`.
        primitive_element
            Optionally specify the primitive element :math:`\alpha` whose powers are roots of the generator polynomial :math:`g(x)`.
            The default is `None` which uses the lexicographically-minimal primitive element in :math:`\mathrm{GF}(2^m)`, see
            :func:`~galois.primitive_element`.
        systematic
            Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
            appended. The default is `True`.

        See Also
        --------
        bch_valid_codes, primitive_poly, primitive_element
        """
        # NOTE: All other arguments will be verified in `_check_and_compute_field()`
        verify_isinstance(systematic, bool)

        self._n = n
        self._k = k
        self._is_systematic = systematic

        c = 1
        GF = _check_and_compute_field(n, k, c, primitive_poly, primitive_element)
        alpha = GF.primitive_element
        m = GF.degree

        t = int(math.ceil((n - k) / m))  # The minimum value of t
        found = False
        while True:
            # We want to find LCM(m_r1(x), m_r2(x), ...) with ri being an element of `roots_`. Instead of computing each
            # minimal polynomial and then doing an LCM, we will compute all the unique conjugates of all the roots
            # and then compute (x - c1)*(x - c2)*...*(x - cn), which is equivalent.
            roots = alpha**(c + np.arange(0, 2*t))
            powers = GF.characteristic**np.arange(0, GF.degree, dtype=GF.dtypes[-1])
            conjugates = np.unique(np.power.outer(roots, powers))
            g_degree = len(conjugates)

            if g_degree < n - k:
                # This t is too small to produce the BCH code
                t += 1
            elif g_degree == n - k:
                # This t produces the correct BCH code size and g(x) is its generator, but there may be a larger t, so keep looking
                found = True
                largest_t_roots = roots
                largest_t_conjugates = conjugates
                t += 1
            elif found and g_degree > n - k:
                # This t does not produce a valid code, but the previous t (which is the largest) did, use it
                break
            else:
                raise ValueError(f"The code BCH({n}, {k}) with c={c} does not exist.")

        g = Poly.Roots(largest_t_conjugates)  # Compute the generator polynomial in GF(2^m)
        g = Poly(g.coeffs, field=GF2)  # Convert coefficients from GF(2^m) to GF(2)

        self._generator_poly = g
        self._roots = largest_t_roots
        self._field = GF
        self._t = self.roots.size // 2

        self._G = poly_to_generator_matrix(n, self.generator_poly, systematic)
        self._H = roots_to_parity_check_matrix(n, self.roots)

        self._is_primitive = True
        self._is_narrow_sense = True

    def __repr__(self) -> str:
        """
        A terse representation of the BCH code.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7)
            bch
        """
        return f"<BCH Code: [{self.n}, {self.k}, {self.d}] over GF(2)>"

    def __str__(self) -> str:
        """
        A formatted string with relevant properties of the BCH code.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7)
            print(bch)
        """
        string = "BCH Code:"
        string += f"\n  [n, k, d]: [{self.n}, {self.k}, {self.d}]"
        string += f"\n  field: {GF2.name}"
        string += f"\n  generator_poly: {self.generator_poly}"
        string += f"\n  is_primitive: {self.is_primitive}"
        string += f"\n  is_narrow_sense: {self.is_narrow_sense}"
        string += f"\n  is_systematic: {self.is_systematic}"
        string += f"\n  t: {self.t}"

        return string

    def encode(self, message: ArrayLike, parity_only: bool = False) -> GF2:
        r"""
        Encodes the message :math:`\mathbf{m}` into the BCH codeword :math:`\mathbf{c}`.

        Parameters
        ----------
        message
            The message as either a :math:`k`-length vector or :math:`(N, k)` matrix, where :math:`N` is the number
            of messages. For systematic codes, message lengths less than :math:`k` may be provided to produce
            shortened codewords.
        parity_only
            Optionally specify whether to return only the parity bits. This only applies to systematic codes.
            The default is `False`.

        Returns
        -------
        :
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix. If `parity_only=True`, the parity
            bits are returned as either a :math:`n - k`-length vector or :math:`(N, n-k)` matrix.

        Notes
        -----
        The message vector :math:`\mathbf{m}` is defined as :math:`\mathbf{m} = [m_{k-1}, \dots, m_1, m_0] \in \mathrm{GF}(2)^k`,
        which corresponds to the message polynomial :math:`m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0`. The codeword vector :math:`\mathbf{c}`
        is defined as :math:`\mathbf{c} = [c_{n-1}, \dots, c_1, c_0] \in \mathrm{GF}(2)^n`, which corresponds to the codeword
        polynomial :math:`c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0`.

        The codeword vector is computed from the message vector by :math:`\mathbf{c} = \mathbf{m}\mathbf{G}`, where :math:`\mathbf{G}` is the
        generator matrix. The equivalent polynomial operation is :math:`c(x) = m(x)g(x)`. For systematic codes, :math:`\mathbf{G} = [\mathbf{I}\ |\ \mathbf{P}]`
        such that :math:`\mathbf{c} = [\mathbf{m}\ |\ \mathbf{p}]`. And in polynomial form, :math:`p(x) = -(m(x) x^{n-k}\ \textrm{mod}\ g(x))` with
        :math:`c(x) = m(x)x^{n-k} + p(x)`. For systematic and non-systematic codes, each codeword is a multiple of the generator polynomial, i.e.
        :math:`g(x)\ |\ c(x)`.

        For the shortened :math:`\textrm{BCH}(n-s, k-s)` code (only applicable for systematic codes), pass :math:`k-s` bits into
        :func:`encode` to return the :math:`n-s`-bit codeword.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Vector

                Encode a single message using the :math:`\textrm{BCH}(15, 7)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random(bch.k); m
                    c = bch.encode(m); c

                Compute the parity bits only.

                .. ipython:: python

                    p = bch.encode(m, parity_only=True); p

            .. tab-item:: Vector (shortened)

                Encode a single message using the shortened :math:`\textrm{BCH}(12, 4)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random(bch.k - 3); m
                    c = bch.encode(m); c

                Compute the parity bits only.

                .. ipython:: python

                    p = bch.encode(m, parity_only=True); p

            .. tab-item:: Matrix

                Encode a matrix of three messages using the :math:`\textrm{BCH}(15, 7)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random((3, bch.k)); m
                    c = bch.encode(m); c

                Compute the parity bits only.

                .. ipython:: python

                    p = bch.encode(m, parity_only=True); p

            .. tab-item:: Matrix (shortened)

                Encode a matrix of three messages using the shortened :math:`\textrm{BCH}(12, 4)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random((3, bch.k - 3)); m
                    c = bch.encode(m); c

                Compute the parity bits only.

                .. ipython:: python

                    p = bch.encode(m, parity_only=True); p
        """
        message = GF2(message)  # This performs type/value checking
        if parity_only and not self.is_systematic:
            raise ValueError("Argument `parity_only=True` only applies to systematic codes.")
        if self.is_systematic:
            if not message.shape[-1] <= self.k:
                raise ValueError(f"For a systematic code, argument `message` must be a 1-D or 2-D array with last dimension less than or equal to {self.k}, not shape {message.shape}.")
        else:
            if not message.shape[-1] == self.k:
                raise ValueError(f"For a non-systematic code, argument `message` must be a 1-D or 2-D array with last dimension equal to {self.k}, not shape {message.shape}.")

        ks = message.shape[-1]  # The number of input message bits (could be less than self.k for shortened codes)

        if parity_only:
            parity = message @ self.G[-ks:, self.k:]
            return parity
        elif self.is_systematic:
            parity = message @ self.G[-ks:, self.k:]
            codeword = np.hstack((message, parity))
            return codeword
        else:
            codeword = message @ self.G
            return codeword

    def detect(self, codeword: ArrayLike) -> Union[np.bool_, np.ndarray]:
        r"""
        Detects if errors are present in the BCH codeword :math:`\mathbf{c}`.

        The :math:`[n, k, d]_2` BCH code has :math:`d_{min} \ge d` minimum distance. It can detect up
        to :math:`d_{min}-1` errors.

        Parameters
        ----------
        codeword
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix, where :math:`N` is the
            number of codewords. For systematic codes, codeword lengths less than :math:`n` may be provided for
            shortened codewords.

        Returns
        -------
        :
            A boolean scalar or array indicating if errors were detected in the corresponding codeword `True` or not `False`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Vector

                Encode a single message using the :math:`\textrm{BCH}(15, 7)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random(bch.k); m
                    c = bch.encode(m); c

                Detect no errors in the valid codeword.

                .. ipython:: python

                    bch.detect(c)

                Detect :math:`d_{min}-1` errors in the codeword.

                .. ipython:: python

                    bch.d
                    c[0:bch.d - 1] ^= 1; c
                    bch.detect(c)

            .. tab-item:: Vector (shortened)

                Encode a single message using the shortened :math:`\textrm{BCH}(12, 4)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random(bch.k - 3); m
                    c = bch.encode(m); c

                Detect no errors in the valid codeword.

                .. ipython:: python

                    bch.detect(c)

                Detect :math:`d_{min}-1` errors in the codeword.

                .. ipython:: python

                    bch.d
                    c[0:bch.d - 1] ^= 1; c
                    bch.detect(c)

            .. tab-item:: Matrix

                Encode a matrix of three messages using the :math:`\textrm{BCH}(15, 7)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random((3, bch.k)); m
                    c = bch.encode(m); c

                Detect no errors in the valid codewords.

                .. ipython:: python

                    bch.detect(c)

                Detect one, two, and :math:`d_{min}-1` errors in the codewords.

                .. ipython:: python

                    bch.d
                    c[0,0:1] ^= 1
                    c[1,0:2] ^= 1
                    c[2, 0:bch.d - 1] ^= 1
                    c
                    bch.detect(c)

            .. tab-item:: Matrix (shortened)

                Encode a matrix of three messages using the shortened :math:`\textrm{BCH}(12, 4)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random((3, bch.k - 3)); m
                    c = bch.encode(m); c

                Detect no errors in the valid codewords.

                .. ipython:: python

                    bch.detect(c)

                Detect one, two, and :math:`d_{min}-1` errors in the codewords.

                .. ipython:: python

                    bch.d
                    c[0,0:1] ^= 1
                    c[1,0:2] ^= 1
                    c[2, 0:bch.d - 1] ^= 1
                    c
                    bch.detect(c)
        """
        codeword = GF2(codeword)  # This performs type/value checking
        if self.is_systematic:
            if not codeword.shape[-1] <= self.n:
                raise ValueError(f"For a systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension less than or equal to {self.n}, not shape {codeword.shape}.")
        else:
            if not codeword.shape[-1] == self.n:
                raise ValueError(f"For a non-systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension equal to {self.n}, not shape {codeword.shape}.")

        codeword_1d = codeword.ndim == 1
        ns = codeword.shape[-1]  # The number of input codeword bits (could be less than self.n for shortened codes)

        # Make codeword 2-D for array processing
        codeword = np.atleast_2d(codeword)

        # Compute the syndrome by matrix multiplying with the parity-check matrix
        syndrome = codeword.view(self.field) @ self.H[:,-ns:].T

        detected = ~np.all(syndrome == 0, axis=1)

        if codeword_1d:
            detected = detected[0]

        return detected

    @overload
    def decode(self, codeword: ArrayLike, errors: Literal[False] = False) -> GF2:
        ...
    @overload
    def decode(self, codeword: ArrayLike, errors: Literal[True]) -> Tuple[GF2, Union[np.integer, np.ndarray]]:
        ...
    def decode(self, codeword, errors=False):
        r"""
        Decodes the BCH codeword :math:`\mathbf{c}` into the message :math:`\mathbf{m}`.

        Parameters
        ----------
        codeword
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix, where :math:`N` is the
            number of codewords. For systematic codes, codeword lengths less than :math:`n` may be provided for
            shortened codewords.
        errors
            Optionally specify whether to return the number of corrected errors. The default is `False`.

        Returns
        -------
        :
            The decoded message as either a :math:`k`-length vector or :math:`(N, k)` matrix.
        :
            Optional return argument of the number of corrected bit errors as either a scalar or :math:`n`-length vector.
            Valid number of corrections are in :math:`[0, t]`. If a codeword has too many errors and cannot be corrected,
            -1 will be returned.

        Notes
        -----
        The codeword vector :math:`\mathbf{c}` is defined as :math:`\mathbf{c} = [c_{n-1}, \dots, c_1, c_0] \in \mathrm{GF}(2)^n`,
        which corresponds to the codeword polynomial :math:`c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0`. The message vector :math:`\mathbf{m}`
        is defined as :math:`\mathbf{m} = [m_{k-1}, \dots, m_1, m_0] \in \mathrm{GF}(2)^k`, which corresponds to the message
        polynomial :math:`m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0`.

        In decoding, the syndrome vector :math:`s` is computed by :math:`\mathbf{s} = \mathbf{c}\mathbf{H}^T`, where
        :math:`\mathbf{H}` is the parity-check matrix. The equivalent polynomial operation is :math:`s(x) = c(x)\ \textrm{mod}\ g(x)`.
        A syndrome of zeros indicates the received codeword is a valid codeword and there are no errors. If the syndrome is non-zero,
        the decoder will find an error-locator polynomial :math:`\sigma(x)` and the corresponding error locations and values.

        For the shortened :math:`\textrm{BCH}(n-s, k-s)` code (only applicable for systematic codes), pass :math:`n-s` bits into
        :func:`decode` to return the :math:`k-s`-bit message.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Vector

                Encode a single message using the :math:`\textrm{BCH}(15, 7)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random(bch.k); m
                    c = bch.encode(m); c

                Corrupt :math:`t` bits of the codeword.

                .. ipython:: python

                    bch.t
                    c[0:bch.t] ^= 1; c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = bch.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = bch.decode(c, errors=True); d, e
                    np.array_equal(d, m)

            .. tab-item:: Vector (shortened)

                Encode a single message using the shortened :math:`\textrm{BCH}(12, 4)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random(bch.k - 3); m
                    c = bch.encode(m); c

                Corrupt :math:`t` bits of the codeword.

                .. ipython:: python

                    bch.t
                    c[0:bch.t] ^= 1; c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = bch.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = bch.decode(c, errors=True); d, e
                    np.array_equal(d, m)

            .. tab-item:: Matrix

                Encode a matrix of three messages using the :math:`\textrm{BCH}(15, 7)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random((3, bch.k)); m
                    c = bch.encode(m); c

                Corrupt the codeword. Add zero errors to the first codeword, one to the second, and two to the third.

                .. ipython:: python

                    c[1,0:1] ^= 1
                    c[2,0:2] ^= 1
                    c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = bch.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = bch.decode(c, errors=True); d, e
                    np.array_equal(d, m)

            .. tab-item:: Matrix (shortened)

                Encode a matrix of three messages using the shortened :math:`\textrm{BCH}(12, 4)` code.

                .. ipython:: python

                    bch = galois.BCH(15, 7)
                    GF = galois.GF(2)
                    m = GF.Random((3, bch.k - 3)); m
                    c = bch.encode(m); c

                Corrupt the codeword. Add zero errors to the first codeword, one to the second, and two to the third.

                .. ipython:: python

                    c[1,0:1] ^= 1
                    c[2,0:2] ^= 1
                    c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = bch.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = bch.decode(c, errors=True); d, e
                    np.array_equal(d, m)
        """
        codeword = GF2(codeword)  # This performs type/value checking
        if self.is_systematic:
            if not codeword.shape[-1] <= self.n:
                raise ValueError(f"For a systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension less than or equal to {self.n}, not shape {codeword.shape}.")
        else:
            if not codeword.shape[-1] == self.n:
                raise ValueError(f"For a non-systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension equal to {self.n}, not shape {codeword.shape}.")

        codeword_1d = codeword.ndim == 1
        ns = codeword.shape[-1]  # The number of input codeword bits (could be less than self.n for shortened codes)
        ks = self.k - (self.n - ns)  # The equivalent number of input message bits (could be less than self.k for shortened codes)

        # Make codeword 2-D for array processing
        codeword = np.atleast_2d(codeword)

        # Compute the syndrome by matrix multiplying with the parity-check matrix
        syndrome = codeword.view(self.field) @ self.H[:,-ns:].T

        # Invoke the JIT compiled function
        dec_codeword, N_errors = decode_jit(self.field)(codeword, syndrome, self.t, int(self.field.primitive_element))

        if self.is_systematic:
            message = dec_codeword[:, 0:ks]
        else:
            message, _ = divmod_jit(GF2)(dec_codeword[:, 0:ns].view(GF2), self.generator_poly.coeffs)
        message = message.view(GF2)

        if codeword_1d:
            message, N_errors = message[0,:], N_errors[0]

        if not errors:
            return message
        else:
            return message, N_errors

    @property
    def field(self) -> Type[FieldArray]:
        r"""
        The :obj:`~galois.FieldArray` subclass for the :math:`\mathrm{GF}(2^m)` field that defines the BCH code.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.field
            print(bch.field)
        """
        return self._field

    @property
    def n(self) -> int:
        """
        The codeword size :math:`n` of the :math:`[n, k, d]_2` code

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.n
        """
        return self._n

    @property
    def k(self) -> int:
        """
        The message size :math:`k` of the :math:`[n, k, d]_2` code

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.k
        """
        return self._k

    @property
    def d(self) -> int:
        r"""
        The design distance :math:`d` of the :math:`[n, k, d]_2` code. The minimum distance of a BCH code
        may be greater than the design distance, :math:`d_{min} \ge d`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.d
        """
        return 2*self.t + 1

    @property
    def t(self) -> int:
        """
        The error-correcting capability of the code. The code can correct :math:`t` bit errors in a codeword.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.t
        """
        return self._t

    @property
    def is_systematic(self) -> bool:
        """
        Indicates if the code is configured to return codewords in systematic form.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.is_systematic
        """
        return self._is_systematic

    @property
    def generator_poly(self) -> Poly:
        """
        The generator polynomial :math:`g(x)` whose roots are :obj:`roots`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.generator_poly
            # Evaluate the generator polynomial at its roots in GF(2^m)
            bch.generator_poly(bch.roots, field=bch.field)
        """
        return self._generator_poly

    @property
    def roots(self) -> FieldArray:
        r"""
        The :math:`2t` roots of the generator polynomial. These are consecutive powers of :math:`\alpha`, specifically
        :math:`\alpha, \alpha^2, \dots, \alpha^{2t}`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.roots
            # Evaluate the generator polynomial at its roots in GF(2^m)
            bch.generator_poly(bch.roots, field=bch.field)
        """
        return self._roots

    @property
    def G(self) -> GF2:
        r"""
        The generator matrix :math:`\mathbf{G}` with shape :math:`(k, n)`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.G
        """
        return self._G

    @property
    def H(self) -> FieldArray:
        r"""
        The parity-check matrix :math:`\mathbf{H}` with shape :math:`(2t, n)`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.H
        """
        return self._H

    @property
    def is_primitive(self) -> bool:
        """
        Indicates if the BCH code is primitive, meaning :math:`n = 2^m - 1`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.is_primitive
        """
        return self._is_primitive

    @property
    def is_narrow_sense(self) -> bool:
        r"""
        Indicates if the BCH code is narrow sense, meaning the roots of the generator polynomial are consecutive
        powers of :math:`\alpha` starting at 1, i.e. :math:`\alpha, \alpha^2, \dots, \alpha^{2t}`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.is_narrow_sense
            bch.roots
            bch.field.primitive_element**(np.arange(1, 2*bch.t + 1))
        """
        return self._is_narrow_sense


class decode_jit(Function):
    """
    Performs BCH decoding.

    References
    ----------
    * Lin, S. and Costello, D. Error Control Coding. Section 7.4.
    """
    def __call__(self, codeword, syndrome, t, primitive_element):
        if self.field.ufunc_mode != "python-calculate":
            y = self.jit(codeword.astype(np.int64), syndrome.astype(np.int64), t, primitive_element)
        else:
            y = self.python(codeword.view(np.ndarray), syndrome.view(np.ndarray), t, primitive_element)

        dec_codeword, N_errors = y[:,0:-1], y[:,-1]
        dec_codeword = dec_codeword.astype(codeword.dtype)
        dec_codeword = dec_codeword.view(self.field)

        return dec_codeword, N_errors

    def set_globals(self):
        # pylint: disable=global-variable-undefined
        global POLY_ROOTS, BERLEKAMP_MASSEY
        POLY_ROOTS = roots_jit(self.field).function
        BERLEKAMP_MASSEY = berlekamp_massey_jit(self.field).function

    _SIGNATURE = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], int64, int64))

    @staticmethod
    def implementation(codeword, syndrome, t, primitive_element):  # pragma: no cover
        dtype = codeword.dtype
        N = codeword.shape[0]  # The number of codewords
        n = codeword.shape[1]  # The codeword size (could be less than the design n for shortened codes)

        # The last column of the returned decoded codeword is the number of corrected errors
        dec_codeword = np.zeros((N, n + 1), dtype=dtype)
        dec_codeword[:, 0:n] = codeword[:,:]

        for i in range(N):
            if not np.all(syndrome[i,:] == 0):
                # The syndrome vector is S = [S0, S1, ..., S2t-1]

                # The error pattern is defined as the polynomial e(x) = e_j1*x^j1 + e_j2*x^j2 + ... for j1 to jv,
                # implying there are v errors. And δi = e_ji is the i-th error value and βi = α^ji is the i-th error-locator
                # value and ji is the error location.

                # The error-locator polynomial σ(x) = (1 - β1*x)(1 - β2*x)...(1 - βv*x) where βi are the inverse of the roots
                # of σ(x).

                # Compute the error-locator polynomial's v-reversal σ(x^-v), since the syndrome is passed in backwards
                # TODO: Re-evaluate these equations since changing BMA to return characteristic polynomial, not feedback polynomial
                sigma_rev = BERLEKAMP_MASSEY(syndrome[i,::-1])[::-1]
                v = sigma_rev.size - 1  # The number of errors

                if v > t:
                    dec_codeword[i, -1] = -1
                    continue

                # Compute βi, the roots of σ(x^-v) which are the inverse roots of σ(x)
                degrees = np.arange(sigma_rev.size - 1, -1, -1)
                results = POLY_ROOTS(degrees, sigma_rev, primitive_element)
                beta = results[0,:]  # The roots of σ(x^-v)
                error_locations = results[1,:]  # The roots as powers of the primitive element α

                if np.any(error_locations > n - 1):
                    # Indicates there are "errors" in the zero-ed portion of a shortened code, which indicates there are actually
                    # more errors than alleged. Return failure to decode.
                    dec_codeword[i, -1] = -1
                    continue

                if beta.size != v:
                    dec_codeword[i, -1] = -1
                    continue

                for j in range(v):
                    # δi can only be 1
                    dec_codeword[i, n - 1 - error_locations[j]] ^= 1
                dec_codeword[i, -1] = v  # The number of corrected errors

        return dec_codeword
