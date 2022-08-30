"""
A module containing arbitrary Reed-Solomon (RS) codes.
"""
from __future__ import annotations

from typing import Tuple, Optional, Union, Type, overload
from typing_extensions import Literal

import numba
from numba import int64
import numpy as np

from .._domains._function import Function
from .._fields import Field, FieldArray
from .._helper import export, verify_isinstance
from .._lfsr import berlekamp_massey_jit
from .._polys import Poly, matlab_primitive_poly
from .._polys._dense import divmod_jit, roots_jit, evaluate_elementwise_jit
from .._prime import factors
from ..typing import ArrayLike, PolyLike

from ._cyclic import poly_to_generator_matrix, roots_to_parity_check_matrix


@export
class ReedSolomon:
    r"""
    A general :math:`\textrm{RS}(n, k)` code.

    A :math:`\textrm{RS}(n, k)` code is a :math:`[n, k, d]_q` linear block code with codeword size :math:`n`, message
    size :math:`k`, minimum distance :math:`d`, and symbols taken from an alphabet of size :math:`q` (a prime power).

    To create the shortened :math:`\textrm{RS}(n-s, k-s)` code, construct the full-sized :math:`\textrm{RS}(n, k)` code
    and then pass :math:`k-s` symbols into :func:`encode` and :math:`n-s` symbols into :func:`decode()`. Shortened codes are only
    applicable for systematic codes.

    Examples
    --------
    Construct the Reed-Solomon code.

    .. ipython:: python

        rs = galois.ReedSolomon(15, 9)
        GF = rs.field

    Encode a message.

    .. ipython:: python

        m = GF.Random(rs.k); m
        c = rs.encode(m); c

    Corrupt the codeword and decode the message.

    .. ipython:: python

        # Corrupt the first symbol in the codeword
        c[0] ^= 13
        dec_m = rs.decode(c); dec_m
        np.array_equal(dec_m, m)

    .. ipython:: python

        # Instruct the decoder to return the number of corrected symbol errors
        dec_m, N = rs.decode(c, errors=True); dec_m, N
        np.array_equal(dec_m, m)

    :group: fec
    """
    # pylint: disable=no-member

    def __init__(
        self,
        n: int,
        k: int,
        c: int = 1,
        primitive_poly: Optional[PolyLike] = None,
        primitive_element: Optional[PolyLike] = None,
        systematic: bool = True
    ):
        r"""
        Constructs a general :math:`\textrm{RS}(n, k)` code.

        Parameters
        ----------
        n
            The codeword size :math:`n`, must be :math:`n = q - 1` where :math:`q` is a prime power.
        k
            The message size :math:`k`. The error-correcting capability :math:`t` is defined by :math:`n - k = 2t`.
        c
            The first consecutive power of :math:`\alpha`. The default is 1.
        primitive_poly
            Optionally specify the primitive polynomial that defines the extension field :math:`\mathrm{GF}(q)`. The default is
            `None` which uses Matlab's default, see :func:`~galois.matlab_primitive_poly`.
        primitive_element
            Optionally specify the primitive element :math:`\alpha` of :math:`\mathrm{GF}(q)` whose powers are roots of the generator polynomial :math:`g(x)`.
            The default is `None` which uses the lexicographically-minimal primitive element in :math:`\mathrm{GF}(q)`, see
            :func:`~galois.primitive_element`.
        systematic
            Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
            appended. The default is `True`.

        See Also
        --------
        primitive_poly, primitive_element
        """
        verify_isinstance(n, int)
        verify_isinstance(k, int)
        verify_isinstance(c, int)
        verify_isinstance(systematic, bool)

        if not (n - k) % 2 == 0:
            raise ValueError("Arguments `n - k` must be even.")
        p, m = factors(n + 1)
        if not (len(p) == 1 and len(m) == 1):
            raise ValueError(f"Argument `n` must have value `q - 1` for a prime power `q`, not {n}.")
        if not c >= 1:
            raise ValueError(f"Argument `c` must be at least 1, not {c}.")
        p, m = p[0], m[0]

        if primitive_poly is None and m > 1:
            primitive_poly = matlab_primitive_poly(p, m)

        self._n = n
        self._k = k
        self._c = c
        self._is_systematic = systematic

        GF = Field(p**m, irreducible_poly=primitive_poly, primitive_element=primitive_element)
        alpha = GF.primitive_element
        t = (n - k) // 2
        roots = alpha**(c + np.arange(0, 2*t))
        g = Poly.Roots(roots)

        self._generator_poly = g
        self._roots = roots
        self._field = GF
        self._t = self.roots.size // 2

        self._G = poly_to_generator_matrix(n, self.generator_poly, systematic)
        self._H = roots_to_parity_check_matrix(n, self.roots)

        self._is_narrow_sense = c == 1

    def __repr__(self) -> str:
        """
        A terse representation of the Reed-Solomon code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9)
            rs
        """
        return f"<Reed-Solomon Code: [{self.n}, {self.k}, {self.d}] over {self.field.name}>"

    def __str__(self) -> str:
        """
        A formatted string with relevant properties of the Reed-Solomon code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9)
            print(rs)
        """
        string = "Reed-Solomon Code:"
        string += f"\n  [n, k, d]: [{self.n}, {self.k}, {self.d}]"
        string += f"\n  field: {self.field.name}"
        string += f"\n  generator_poly: {self.generator_poly}"
        string += f"\n  is_narrow_sense: {self.is_narrow_sense}"
        string += f"\n  is_systematic: {self.is_systematic}"
        string += f"\n  t: {self.t}"

        return string

    def encode(self, message: ArrayLike, parity_only: bool = False) -> FieldArray:
        r"""
        Encodes the message :math:`\mathbf{m}` into the Reed-Solomon codeword :math:`\mathbf{c}`.

        Parameters
        ----------
        message
            The message as either a :math:`k`-length vector or :math:`(N, k)` matrix, where :math:`N` is the number
            of messages. For systematic codes, message lengths less than :math:`k` may be provided to produce
            shortened codewords.
        parity_only
            Optionally specify whether to return only the parity symbols. This only applies to systematic codes.
            The default is `False`.

        Returns
        -------
        :
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix. If `parity_only=True`, the parity
            symbols are returned as either a :math:`n - k`-length vector or :math:`(N, n-k)` matrix.

        Notes
        -----
        The message vector :math:`\mathbf{m}` is defined as :math:`\mathbf{m} = [m_{k-1}, \dots, m_1, m_0] \in \mathrm{GF}(q)^k`,
        which corresponds to the message polynomial :math:`m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0`. The codeword vector :math:`\mathbf{c}`
        is defined as :math:`\mathbf{c} = [c_{n-1}, \dots, c_1, c_0] \in \mathrm{GF}(q)^n`, which corresponds to the codeword
        polynomial :math:`c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0`.

        The codeword vector is computed from the message vector by :math:`\mathbf{c} = \mathbf{m}\mathbf{G}`, where :math:`\mathbf{G}` is the
        generator matrix. The equivalent polynomial operation is :math:`c(x) = m(x)g(x)`. For systematic codes, :math:`\mathbf{G} = [\mathbf{I}\ |\ \mathbf{P}]`
        such that :math:`\mathbf{c} = [\mathbf{m}\ |\ \mathbf{p}]`. And in polynomial form, :math:`p(x) = -(m(x) x^{n-k}\ \textrm{mod}\ g(x))` with
        :math:`c(x) = m(x)x^{n-k} + p(x)`. For systematic and non-systematic codes, each codeword is a multiple of the generator polynomial, i.e.
        :math:`g(x)\ |\ c(x)`.

        For the shortened :math:`\textrm{RS}(n-s, k-s)` code (only applicable for systematic codes), pass :math:`k-s` symbols into
        :func:`encode` to return the :math:`n-s`-symbol codeword.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Vector

                Encode a single message using the :math:`\textrm{RS}(15, 9)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random(rs.k); m
                    c = rs.encode(m); c

                Compute the parity symbols only.

                .. ipython:: python

                    p = rs.encode(m, parity_only=True); p

            .. tab-item:: Vector (shortened)

                Encode a single message using the shortened :math:`\textrm{RS}(11, 5)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random(rs.k - 4); m
                    c = rs.encode(m); c

                Compute the parity symbols only.

                .. ipython:: python

                    p = rs.encode(m, parity_only=True); p

            .. tab-item:: Matrix

                Encode a matrix of three messages using the :math:`\textrm{RS}(15, 9)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random((3, rs.k)); m
                    c = rs.encode(m); c

                Compute the parity symbols only.

                .. ipython:: python

                    p = rs.encode(m, parity_only=True); p

            .. tab-item:: Matrix (shortened)

                Encode a matrix of three messages using the shortened :math:`\textrm{RS}(11, 5)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random((3, rs.k - 4)); m
                    c = rs.encode(m); c

                Compute the parity symbols only.

                .. ipython:: python

                    p = rs.encode(m, parity_only=True); p
        """
        message = self.field(message)  # This performs type/value checking
        if parity_only and not self.is_systematic:
            raise ValueError("Argument `parity_only=True` only applies to systematic codes.")
        if self.is_systematic:
            if not message.shape[-1] <= self.k:
                raise ValueError(f"For a systematic code, argument `message` must be a 1-D or 2-D array with last dimension less than or equal to {self.k}, not shape {message.shape}.")
        else:
            if not message.shape[-1] == self.k:
                raise ValueError(f"For a non-systematic code, argument `message` must be a 1-D or 2-D array with last dimension equal to {self.k}, not shape {message.shape}.")

        ks = message.shape[-1]  # The number of input message symbols (could be less than self.k for shortened codes)

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
        Detects if errors are present in the Reed-Solomon codeword :math:`\mathbf{c}`.

        The :math:`[n, k, d]_q` Reed-Solomon code has :math:`d_{min} = d` minimum distance. It can detect up
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

                Encode a single message using the :math:`\textrm{RS}(15, 9)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random(rs.k); m
                    c = rs.encode(m); c

                Detect no errors in the valid codeword.

                .. ipython:: python

                    rs.detect(c)

                Detect :math:`d_{min}-1` errors in the codeword.

                .. ipython:: python

                    rs.d
                    e = GF.Random(rs.d - 1, low=1); e
                    c[0:rs.d - 1] += e; c
                    rs.detect(c)

            .. tab-item:: Vector (shortened)

                Encode a single message using the shortened :math:`\textrm{RS}(11, 5)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random(rs.k - 4); m
                    c = rs.encode(m); c

                Detect no errors in the valid codeword.

                .. ipython:: python

                    rs.detect(c)

                Detect :math:`d_{min}-1` errors in the codeword.

                .. ipython:: python

                    rs.d
                    e = GF.Random(rs.d - 1, low=1); e
                    c[0:rs.d - 1] += e; c
                    rs.detect(c)

            .. tab-item:: Matrix

                Encode a matrix of three messages using the :math:`\textrm{RS}(15, 9)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random((3, rs.k)); m
                    c = rs.encode(m); c

                Detect no errors in the valid codewords.

                .. ipython:: python

                    rs.detect(c)

                Detect one, two, and :math:`d_{min}-1` errors in the codewords.

                .. ipython:: python

                    rs.d
                    c[0,0:1] += GF.Random(1, low=1)
                    c[1,0:2] += GF.Random(2, low=1)
                    c[2, 0:rs.d - 1] += GF.Random(rs.d - 1, low=1)
                    c
                    rs.detect(c)

            .. tab-item:: Matrix (shortened)

                Encode a matrix of three messages using the shortened :math:`\textrm{RS}(11, 5)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random((3, rs.k - 4)); m
                    c = rs.encode(m); c

                Detect no errors in the valid codewords.

                .. ipython:: python

                    rs.detect(c)

                Detect one, two, and :math:`d_{min}-1` errors in the codewords.

                .. ipython:: python

                    rs.d
                    c[0,0:1] += GF.Random(1, low=1)
                    c[1,0:2] += GF.Random(2, low=1)
                    c[2, 0:rs.d - 1] += GF.Random(rs.d - 1, low=1)
                    c
                    rs.detect(c)
        """
        codeword = self.field(codeword)  # This performs type/value checking
        if self.is_systematic:
            if not codeword.shape[-1] <= self.n:
                raise ValueError(f"For a systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension less than or equal to {self.n}, not shape {codeword.shape}.")
        else:
            if not codeword.shape[-1] == self.n:
                raise ValueError(f"For a non-systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension equal to {self.n}, not shape {codeword.shape}.")

        codeword_1d = codeword.ndim == 1
        ns = codeword.shape[-1]  # The number of input codeword symbols (could be less than self.n for shortened codes)

        # Make codeword 2-D for array processing
        codeword = np.atleast_2d(codeword)

        # Compute the syndrome by matrix multiplying with the parity-check matrix
        syndrome = codeword.view(self.field) @ self.H[:,-ns:].T

        detected = ~np.all(syndrome == 0, axis=1)

        if codeword_1d:
            detected = detected[0]

        return detected

    @overload
    def decode(self, codeword: ArrayLike, errors: Literal[False] = False) -> FieldArray:
        ...
    @overload
    def decode(self, codeword: ArrayLike, errors: Literal[True]) -> Tuple[FieldArray, Union[np.integer, np.ndarray]]:
        ...
    def decode(self, codeword, errors=False):
        r"""
        Decodes the Reed-Solomon codeword :math:`\mathbf{c}` into the message :math:`\mathbf{m}`.

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
            Optional return argument of the number of corrected symbol errors as either a scalar or :math:`n`-length vector.
            Valid number of corrections are in :math:`[0, t]`. If a codeword has too many errors and cannot be corrected,
            -1 will be returned.

        Notes
        -----
        The codeword vector :math:`\mathbf{c}` is defined as :math:`\mathbf{c} = [c_{n-1}, \dots, c_1, c_0] \in \mathrm{GF}(q)^n`,
        which corresponds to the codeword polynomial :math:`c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0`. The message vector :math:`\mathbf{m}`
        is defined as :math:`\mathbf{m} = [m_{k-1}, \dots, m_1, m_0] \in \mathrm{GF}(q)^k`, which corresponds to the message
        polynomial :math:`m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0`.

        In decoding, the syndrome vector :math:`\mathbf{s}` is computed by :math:`\mathbf{s} = \mathbf{c}\mathbf{H}^T`, where
        :math:`\mathbf{H}` is the parity-check matrix. The equivalent polynomial operation is the codeword polynomial evaluated
        at each root of the generator polynomial, i.e. :math:`\mathbf{s} = [c(\alpha^{c}), c(\alpha^{c+1}), \dots, c(\alpha^{c+2t-1})]`.
        A syndrome of zeros indicates the received codeword is a valid codeword and there are no errors. If the syndrome is non-zero,
        the decoder will find an error-locator polynomial :math:`\sigma(x)` and the corresponding error locations and values.

        For the shortened :math:`\textrm{RS}(n-s, k-s)` code (only applicable for systematic codes), pass :math:`n-s` symbols into
        :func:`decode` to return the :math:`k-s`-symbol message.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Vector

                Encode a single message using the :math:`\textrm{RS}(15, 9)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random(rs.k); m
                    c = rs.encode(m); c

                Corrupt :math:`t` symbols of the codeword.

                .. ipython:: python

                    e = GF.Random(rs.t, low=1); e
                    c[0:rs.t] += e; c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = rs.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = rs.decode(c, errors=True); d, e
                    np.array_equal(d, m)

            .. tab-item:: Vector (shortened)

                Encode a single message using the shortened :math:`\textrm{RS}(11, 5)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random(rs.k - 4); m
                    c = rs.encode(m); c

                Corrupt :math:`t` symbols of the codeword.

                .. ipython:: python

                    e = GF.Random(rs.t, low=1); e
                    c[0:rs.t] += e; c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = rs.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = rs.decode(c, errors=True); d, e
                    np.array_equal(d, m)

            .. tab-item:: Matrix

                Encode a matrix of three messages using the :math:`\textrm{RS}(15, 9)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random((3, rs.k)); m
                    c = rs.encode(m); c

                Corrupt the codeword. Add one error to the first codeword, two to the second, and three to the third.

                .. ipython:: python

                    c[0,0:1] += GF.Random(1, low=1)
                    c[1,0:2] += GF.Random(2, low=1)
                    c[2,0:3] += GF.Random(3, low=1)
                    c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = rs.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = rs.decode(c, errors=True); d, e
                    np.array_equal(d, m)

            .. tab-item:: Matrix (shortened)

                Encode a matrix of three messages using the shortened :math:`\textrm{RS}(11, 5)` code.

                .. ipython:: python

                    rs = galois.ReedSolomon(15, 9)
                    GF = rs.field
                    m = GF.Random((3, rs.k - 4)); m
                    c = rs.encode(m); c

                Corrupt the codeword. Add one error to the first codeword, two to the second, and three to the third.

                .. ipython:: python

                    c[0,0:1] += GF.Random(1, low=1)
                    c[1,0:2] += GF.Random(2, low=1)
                    c[2,0:3] += GF.Random(3, low=1)
                    c

                Decode the codeword and recover the message.

                .. ipython:: python

                    d = rs.decode(c); d
                    np.array_equal(d, m)

                Decode the codeword, specifying the number of corrected errors, and recover the message.

                .. ipython:: python

                    d, e = rs.decode(c, errors=True); d, e
                    np.array_equal(d, m)
        """
        codeword = self.field(codeword)  # This performs type/value checking
        if self.is_systematic:
            if not codeword.shape[-1] <= self.n:
                raise ValueError(f"For a systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension less than or equal to {self.n}, not shape {codeword.shape}.")
        else:
            if not codeword.shape[-1] == self.n:
                raise ValueError(f"For a non-systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension equal to {self.n}, not shape {codeword.shape}.")

        codeword_1d = codeword.ndim == 1
        ns = codeword.shape[-1]  # The number of input codeword symbols (could be less than self.n for shortened codes)
        ks = self.k - (self.n - ns)  # The equivalent number of input message symbols (could be less than self.k for shortened codes)

        # Make codeword 2-D for array processing
        codeword = np.atleast_2d(codeword)

        # Compute the syndrome by matrix multiplying with the parity-check matrix
        syndrome = codeword.view(self.field) @ self.H[:,-ns:].T

        # Invoke the JIT compiled function
        dec_codeword, N_errors = decode_jit(self.field)(codeword, syndrome, self.c, self.t, int(self.field.primitive_element))

        if self.is_systematic:
            message = dec_codeword[:, 0:ks]
        else:
            message, _ = divmod_jit(self.field)(dec_codeword[:, 0:ns].view(self.field), self.generator_poly.coeffs)
        message = message.view(self.field)

        if codeword_1d:
            message, N_errors = message[0,:], N_errors[0]

        if not errors:
            return message
        else:
            return message, N_errors

    @property
    def field(self) -> Type[FieldArray]:
        r"""
        The :obj:`~galois.FieldArray` subclass for the :math:`\mathrm{GF}(q)` field that defines the Reed-Solomon code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.field
            print(rs.field)
        """
        return self._field

    @property
    def n(self) -> int:
        """
        The codeword size :math:`n` of the :math:`[n, k, d]_q` code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.n
        """
        return self._n

    @property
    def k(self) -> int:
        """
        The message size :math:`k` of the :math:`[n, k, d]_q` code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.k
        """
        return self._k

    @property
    def d(self) -> int:
        """
        The design distance :math:`d` of the :math:`[n, k, d]_q` code. The minimum distance of a Reed-Solomon code
        is exactly equal to the design distance, :math:`d_{min} = d`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.d
        """
        return 2*self.t + 1

    @property
    def t(self) -> int:
        """
        The error-correcting capability of the code. The code can correct :math:`t` symbol errors in a codeword.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.t
        """
        return self._t

    @property
    def is_systematic(self) -> bool:
        """
        Indicates if the code is configured to return codewords in systematic form.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.is_systematic
        """
        return self._is_systematic

    @property
    def generator_poly(self) -> Poly:
        """
        The generator polynomial :math:`g(x)` whose roots are :obj:`roots`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.generator_poly

        Evaluate the generator polynomial at its roots.

        .. ipython:: python

            rs.generator_poly(rs.roots)
        """
        return self._generator_poly

    @property
    def roots(self) -> FieldArray:
        r"""
        The :math:`2t` roots of the generator polynomial. These are consecutive powers of :math:`\alpha`, specifically
        :math:`\alpha^c, \alpha^{c+1}, \dots, \alpha^{c+2t-1}`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.roots

        Evaluate the generator polynomial at its roots.

        .. ipython:: python

            rs.generator_poly(rs.roots)
        """
        return self._roots

    @property
    def c(self) -> int:
        """
        The degree of the first consecutive root.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.c
        """
        return self._c

    @property
    def G(self) -> FieldArray:
        r"""
        The generator matrix :math:`\mathbf{G}` with shape :math:`(k, n)`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.G
        """
        return self._G

    @property
    def H(self) -> FieldArray:
        r"""
        The parity-check matrix :math:`\mathbf{H}` with shape :math:`(2t, n)`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.H
        """
        return self._H

    @property
    def is_narrow_sense(self) -> bool:
        r"""
        Indicates if the Reed-Solomon code is narrow sense, meaning the roots of the generator polynomial are consecutive
        powers of :math:`\alpha` starting at 1, i.e. :math:`\alpha, \alpha^2, \dots, \alpha^{2t - 1}`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.is_narrow_sense
            rs.roots
            rs.field.primitive_element**(np.arange(1, 2*rs.t + 1))
        """
        return self._is_narrow_sense


class decode_jit(Function):
    """
    Performs Reed-Solomon decoding.

    References
    ----------
    * Lin, S. and Costello, D. Error Control Coding. Section 7.4.
    """
    def __call__(self, codeword, syndrome, c, t, primitive_element):
        if self.field.ufunc_mode != "python-calculate":
            y = self.jit(codeword.astype(np.int64), syndrome.astype(np.int64), c, t, primitive_element)
        else:
            y = self.python(codeword.view(np.ndarray), syndrome.view(np.ndarray), c, t, primitive_element)

        dec_codeword, N_errors = y[:,0:-1], y[:,-1]
        dec_codeword = dec_codeword.astype(codeword.dtype)
        dec_codeword = dec_codeword.view(self.field)

        return dec_codeword, N_errors

    def set_globals(self):
        # pylint: disable=global-variable-undefined
        global CHARACTERISTIC, ORDER, SUBTRACT, MULTIPLY, RECIPROCAL, POWER, CONVOLVE, POLY_ROOTS, POLY_EVALUATE, BERLEKAMP_MASSEY
        CHARACTERISTIC = self.field.characteristic
        ORDER = self.field.order
        SUBTRACT = self.field._subtract.ufunc
        MULTIPLY = self.field._multiply.ufunc
        RECIPROCAL = self.field._reciprocal.ufunc
        POWER = self.field._power.ufunc
        CONVOLVE = self.field._convolve.function
        POLY_ROOTS = roots_jit(self.field).function
        POLY_EVALUATE = evaluate_elementwise_jit(self.field).function
        BERLEKAMP_MASSEY = berlekamp_massey_jit(self.field).function

    _SIGNATURE = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], int64, int64, int64))

    @staticmethod
    def implementation(codeword, syndrome, c, t, primitive_element):  # pragma: no cover
        dtype = codeword.dtype
        N = codeword.shape[0]  # The number of codewords
        n = codeword.shape[1]  # The codeword size (could be less than the design n for shortened codes)
        design_n = ORDER - 1  # The designed codeword size

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

                # Compute the error-locator polynomial σ(x)
                # TODO: Re-evaluate these equations since changing BMA to return characteristic polynomial, not feedback polynomial
                sigma = BERLEKAMP_MASSEY(syndrome[i,:])[::-1]
                v = sigma.size - 1  # The number of errors, which is the degree of the error-locator polynomial

                if v > t:
                    dec_codeword[i,-1] = -1
                    continue

                # Compute βi^-1, the roots of σ(x)
                degrees = np.arange(sigma.size - 1, -1, -1)
                results = POLY_ROOTS(degrees, sigma, primitive_element)
                beta_inv = results[0,:]  # The roots βi^-1 of σ(x)
                error_locations_inv = results[1,:]  # The roots βi^-1 as powers of the primitive element α
                error_locations = -error_locations_inv % design_n  # The error locations as degrees of c(x)

                if np.any(error_locations > n - 1):
                    # Indicates there are "errors" in the zero-ed portion of a shortened code, which indicates there are actually
                    # more errors than alleged. Return failure to decode.
                    dec_codeword[i,-1] = -1
                    continue

                if beta_inv.size != v:
                    dec_codeword[i,-1] = -1
                    continue

                # Compute σ'(x)
                sigma_prime = np.zeros(v, dtype=dtype)
                for j in range(v):
                    degree = v - j
                    sigma_prime[j] = MULTIPLY(degree % CHARACTERISTIC, sigma[j])  # Scalar multiplication

                # The error-value evaluator polynomial Z0(x) = S0*σ0 + (S1*σ0 + S0*σ1)*x + (S2*σ0 + S1*σ1 + S0*σ2)*x^2 + ...
                # with degree v-1
                Z0 = CONVOLVE(sigma[-v:], syndrome[i,0:v][::-1])[-v:]

                # The error value δi = -1 * βi^(1-c) * Z0(βi^-1) / σ'(βi^-1)
                for j in range(v):
                    beta_i = POWER(beta_inv[j], c - 1)
                    Z0_i = POLY_EVALUATE(Z0, np.array([beta_inv[j]], dtype=dtype))[0]  # NOTE: poly_eval() expects a 1-D array of values
                    sigma_prime_i = POLY_EVALUATE(sigma_prime, np.array([beta_inv[j]], dtype=dtype))[0]  # NOTE: poly_eval() expects a 1-D array of values
                    delta_i = MULTIPLY(beta_i, Z0_i)
                    delta_i = MULTIPLY(delta_i, RECIPROCAL(sigma_prime_i))
                    delta_i = SUBTRACT(0, delta_i)
                    dec_codeword[i, n - 1 - error_locations[j]] = SUBTRACT(dec_codeword[i, n - 1 - error_locations[j]], delta_i)

                dec_codeword[i,-1] = v  # The number of corrected errors

        return dec_codeword
