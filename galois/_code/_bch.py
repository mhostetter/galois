import math

import numba
from numba import int64
import numpy as np

from .. import _lfsr
from .._factor import factors
from .._field import Field, Poly, GF2, matlab_primitive_poly
from .._field._meta_function import UNARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, POLY_ROOTS_CALCULATE_SIG
from .._overrides import set_module

from ._cyclic import poly_to_generator_matrix, roots_to_parity_check_matrix

__all__ = ["BCH", "bch_valid_codes"]


def _check_and_compute_field(n, k, c, primitive_poly, primitive_element):
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
    if not isinstance(c, (int, np.integer)):
        raise TypeError(f"Argument `c` must be an integer, not {type(c)}.")
    if not isinstance(primitive_poly, (type(None), int, Poly)):
        raise TypeError(f"Argument `primitive_poly` must be an int or galois.Poly, not {type(primitive_poly)}.")
    if not isinstance(primitive_element, (type(None), int, Poly)):
        raise TypeError(f"Argument `primitive_element` must be an int or galois.Poly, not {type(primitive_element)}.")

    p, m = factors(n + 1)
    if not (len(p) == 1 and p[0] == 2):
        raise ValueError(f"Argument `n` must have value `2^m - 1` for some positive m, not {n}.")
    if not c >= 1:
        raise ValueError(f"Argument `c` must be at least 1, not {c}.")
    p, m = p[0], m[0]

    if primitive_poly is None:
        primitive_poly = matlab_primitive_poly(2, m)

    GF = Field(2**m, irreducible_poly=primitive_poly, primitive_element=primitive_element)

    return GF


@set_module("galois")
def bch_valid_codes(n, t_min=1):
    r"""
    Returns a list of :math:`(n, k, t)` tuples of valid primitive binary BCH codes.

    A BCH code with parameters :math:`(n, k, t)` is represented as a :math:`[n, k, d]_2` linear block
    code with :math:`d = 2t + 1`.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = 2^m - 1`.
    t_min : int, optional
        The minimum error-correcting capability. The default is 1.

    Returns
    -------
    list
        A list of :math:`(n, k, t)` tuples of valid primitive BCH codes.

    References
    ----------
    * https://link.springer.com/content/pdf/bbm%3A978-1-4899-2174-1%2F1.pdf

    Examples
    --------
    .. ipython:: python

        galois.bch_valid_codes(31)
        galois.bch_valid_codes(31, t_min=3)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(t_min, (int, np.integer)):
        raise TypeError(f"Argument `t_min` must be an integer, not {type(t_min)}.")
    if not t_min >= 1:
        raise ValueError(f"Argument `t_min` must be at least 1, not {t_min}.")

    GF = _check_and_compute_field(n, 0, 1, None, None)  # NOTE: k isn't needed for generating the field
    alpha = GF.primitive_element

    codes = []
    t = t_min
    while True:
        c = 1
        roots = alpha**(c + np.arange(0, 2*t))
        powers = GF.characteristic**np.arange(0, GF.degree)
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


@set_module("galois")
class BCH:
    r"""
    Constructs a primitive, narrow-sense binary :math:`\textrm{BCH}(n, k)` code.

    A :math:`\textrm{BCH}(n, k)` code is a :math:`[n, k, d]_2` linear block code.

    To create the shortened :math:`\textrm{BCH}(n-s, k-s)` code, construct the full-sized :math:`\textrm{BCH}(n, k)` code
    and then pass :math:`k-s` bits into :func:`encode` and :math:`n-s` bits into :func:`decode()`. Shortened codes are only
    applicable for systematic codes.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = 2^m - 1`.
    k : int
        The message size :math:`k`.
    primitive_poly : galois.Poly, optional
        Optionally specify the primitive polynomial that defines the extension field :math:`\mathrm{GF}(2^m)`. The default is
        `None` which uses Matlab's default, see :func:`galois.matlab_primitive_poly`. Matlab tends to use the lexicographically-minimal
        primitive polynomial as a default instead of the Conway polynomial.
    primitive_element : int, galois.Poly, optional
        Optionally specify the primitive element :math:`\alpha` whose powers are roots of the generator polynomial :math:`g(x)`.
        The default is `None` which uses the lexicographically-minimal primitive element in :math:`\mathrm{GF}(2^m)`, see
        :func:`galois.primitive_element`.
    systematic : bool, optional
        Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
        appended. The default is `True`.

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
    """
    # pylint: disable=no-member

    def __new__(cls, n, k, primitive_poly=None, primitive_element=None, systematic=True):
        # NOTE: All other arguments will be verified in `_check_and_compute_field()`
        if not isinstance(systematic, bool):
            raise TypeError(f"Argument `systematic` must be a bool, not {type(systematic)}.")

        obj = super().__new__(cls)

        obj._n = n
        obj._k = k
        obj._systematic = systematic

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
            powers = GF.characteristic**np.arange(0, GF.degree)
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

        obj._generator_poly = g
        obj._roots = largest_t_roots
        obj._field = GF
        obj._t = obj.roots.size // 2

        obj._G = poly_to_generator_matrix(n, obj.generator_poly, systematic)
        obj._H = roots_to_parity_check_matrix(n, obj.roots)

        obj._is_primitive = True
        obj._is_narrow_sense = True

        return obj

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        # Pre-compile the arithmetic methods
        self._add_jit = self.field._calculate_jit("add")
        self._subtract_jit = self.field._calculate_jit("subtract")
        self._multiply_jit = self.field._calculate_jit("multiply")
        self._reciprocal_jit = self.field._calculate_jit("reciprocal")
        self._power_jit = self.field._calculate_jit("power")

        # Pre-compile the JIT functions
        self._berlekamp_massey_jit = _lfsr.jit_calculate("berlekamp_massey")
        self._poly_roots_jit = self.field._function("poly_roots")
        self._poly_divmod_jit = GF2._function("poly_divmod")

        # Pre-compile the JIT decoder
        self._decode_jit = numba.jit(DECODE_CALCULATE_SIG.signature, nopython=True, cache=True)(decode_calculate)

    def __str__(self):
        return f"<BCH Code: [{self.n}, {self.k}, {self.d}] over GF(2)>"

    def __repr__(self):
        return str(self)

    def encode(self, message, parity_only=False):
        r"""
        Encodes the message :math:`\mathbf{m}` into the BCH codeword :math:`\mathbf{c}`.

        Parameters
        ----------
        message : numpy.ndarray, galois.FieldArray
            The message as either a :math:`k`-length vector or :math:`(N, k)` matrix, where :math:`N` is the number
            of messages. For systematic codes, message lengths less than :math:`k` may be provided to produce
            shortened codewords.
        parity_only : bool, optional
            Optionally specify whether to return only the parity bits. This only applies to systematic codes.
            The default is `False`.

        Returns
        -------
        numpy.ndarray, galois.FieldArray
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix. The return type matches the
            message type. If `parity_only=True`, the parity bits are returned as either a :math:`n - k`-length vector or
            :math:`(N, n-k)` matrix.

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
        Encode a single codeword.

        .. ipython:: python

            bch = galois.BCH(15, 7)
            m = galois.GF2.Random(bch.k); m
            c = bch.encode(m); c
            p = bch.encode(m, parity_only=True); p

        Encode a single, shortened codeword.

        .. ipython:: python

            m = galois.GF2.Random(bch.k - 3); m
            c = bch.encode(m); c

        Encode a matrix of codewords.

        .. ipython:: python

            m = galois.GF2.Random((5, bch.k)); m
            c = bch.encode(m); c
            p = bch.encode(m, parity_only=True); p
        """
        if not isinstance(message, np.ndarray):
            raise TypeError(f"Argument `message` must be a subclass of np.ndarray (or a galois.GF2 array), not {type(message)}.")
        if parity_only and not self.systematic:
            raise ValueError("Argument `parity_only=True` only applies to systematic codes.")
        if self.systematic:
            if not message.shape[-1] <= self.k:
                raise ValueError(f"For a systematic code, argument `message` must be a 1-D or 2-D array with last dimension less than or equal to {self.k}, not shape {message.shape}.")
        else:
            if not message.shape[-1] == self.k:
                raise ValueError(f"For a non-systematic code, argument `message` must be a 1-D or 2-D array with last dimension equal to {self.k}, not shape {message.shape}.")

        ks = message.shape[-1]  # The number of input message bits (could be less than self.k for shortened codes)

        if parity_only:
            parity = message.view(GF2) @ self.G[-ks:, self.k:]
            return parity.view(type(message))
        elif self.systematic:
            parity = message.view(GF2) @ self.G[-ks:, self.k:]
            return np.hstack((message, parity)).view(type(message))
        else:
            codeword = message.view(GF2) @ self.G
            return codeword.view(type(message))

    def detect(self, codeword):
        r"""
        Detects if errors are present in the BCH codeword :math:`\mathbf{c}`.

        The :math:`[n, k, d]_2` BCH code has :math:`d_{min} \ge d` minimum distance. It can detect up
        to :math:`d_{min}-1` errors.

        Parameters
        ----------
        codeword : numpy.ndarray, galois.FieldArray
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix, where :math:`N` is the
            number of codewords. For systematic codes, codeword lengths less than :math:`n` may be provided for
            shortened codewords.

        Returns
        -------
        bool, numpy.ndarray
            A boolean scalar or array indicating if errors were detected in the corresponding codeword `True` or not `False`.

        Examples
        --------
        Detect errors in a valid codeword.

        .. ipython:: python

            bch = galois.BCH(15, 7)
            # The minimum distance of the code
            bch.d
            m = galois.GF2.Random(bch.k); m
            c = bch.encode(m); c
            bch.detect(c)

        Detect :math:`d_{min}-1` errors in a received codeword.

        .. ipython:: python

            # Corrupt the first `d - 1` bits in the codeword
            c[0:bch.d - 1] ^= 1
            bch.detect(c)
        """
        if not isinstance(codeword, np.ndarray):
            raise TypeError(f"Argument `codeword` must be a subclass of np.ndarray (or a galois.GF2 array), not {type(codeword)}.")
        if self.systematic:
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

    def decode(self, codeword, errors=False):
        r"""
        Decodes the BCH codeword :math:`\mathbf{c}` into the message :math:`\mathbf{m}`.

        Parameters
        ----------
        codeword : numpy.ndarray, galois.FieldArray
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix, where :math:`N` is the
            number of codewords. For systematic codes, codeword lengths less than :math:`n` may be provided for
            shortened codewords.
        errors : bool, optional
            Optionally specify whether to return the nubmer of corrected errors.

        Returns
        -------
        numpy.ndarray, galois.FieldArray
            The decoded message as either a :math:`k`-length vector or :math:`(N, k)` matrix.
        int, np.ndarray
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
        Decode a single codeword.

        .. ipython:: python

            bch = galois.BCH(15, 7)
            m = galois.GF2.Random(bch.k); m
            c = bch.encode(m); c
            # Corrupt the first bit in the codeword
            c[0] ^= 1
            dec_m = bch.decode(c); dec_m
            np.array_equal(dec_m, m)

            # Instruct the decoder to return the number of corrected bit errors
            dec_m, N = bch.decode(c, errors=True); dec_m, N
            np.array_equal(dec_m, m)

        Decode a single, shortened codeword.

        .. ipython:: python

            m = galois.GF2.Random(bch.k - 3); m
            c = bch.encode(m); c
            # Corrupt the first bit in the codeword
            c[0] ^= 1
            dec_m = bch.decode(c); dec_m
            np.array_equal(dec_m, m)

        Decode a matrix of codewords.

        .. ipython:: python

            m = galois.GF2.Random((5, bch.k)); m
            c = bch.encode(m); c
            # Corrupt the first bit in each codeword
            c[:,0] ^= 1
            dec_m = bch.decode(c); dec_m
            np.array_equal(dec_m, m)

            # Instruct the decoder to return the number of corrected bit errors
            dec_m, N = bch.decode(c, errors=True); dec_m, N
            np.array_equal(dec_m, m)
        """
        if not isinstance(codeword, np.ndarray):
            raise TypeError(f"Argument `codeword` must be a subclass of np.ndarray (or a galois.GF2 array), not {type(codeword)}.")
        if self.systematic:
            if not codeword.shape[-1] <= self.n:
                raise ValueError(f"For a systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension less than or equal to {self.n}, not shape {codeword.shape}.")
        else:
            if not codeword.shape[-1] == self.n:
                raise ValueError(f"For a non-systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension equal to {self.n}, not shape {codeword.shape}.")

        codeword_1d = codeword.ndim == 1
        dtype = codeword.dtype
        ns = codeword.shape[-1]  # The number of input codeword bits (could be less than self.n for shortened codes)
        ks = self.k - (self.n - ns)  # The equivalent number of input message bits (could be less than self.k for shortened codes)

        # Make codeword 2-D for array processing
        codeword = np.atleast_2d(codeword)

        # Compute the syndrome by matrix multiplying with the parity-check matrix
        syndrome = codeword.view(self.field) @ self.H[:,-ns:].T

        if self.field.ufunc_mode != "python-calculate":
            dec_codeword =  self._decode_jit(codeword.astype(np.int64), syndrome.astype(np.int64), self.t, int(self.field.primitive_element), self._add_jit, self._subtract_jit, self._multiply_jit, self._reciprocal_jit, self._power_jit, self._berlekamp_massey_jit, self._poly_roots_jit, self.field.characteristic, self.field.degree, self.field._irreducible_poly_int)
            N_errors = dec_codeword[:, -1]

            if self.systematic:
                message = dec_codeword[:, 0:ks]
            else:
                message, _ = GF2._poly_divmod(dec_codeword[:, 0:self.n].view(GF2), self.generator_poly.coeffs)
            message = message.astype(dtype).view(type(codeword))

        else:
            raise NotImplementedError("BCH codes haven't been implemented for extremely large Galois fields.")

        if codeword_1d:
            message, N_errors = message[0,:], N_errors[0]

        if not errors:
            return message
        else:
            return message, N_errors

    @property
    def field(self):
        r"""
        galois.FieldClass: The Galois field :math:`\mathrm{GF}(2^m)` that defines the BCH code.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.field
            print(bch.field.properties)
        """
        return self._field

    @property
    def n(self):
        """
        int: The codeword size :math:`n` of the :math:`[n, k, d]_2` code


        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.n
        """
        return self._n

    @property
    def k(self):
        """
        int: The message size :math:`k` of the :math:`[n, k, d]_2` code

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.k
        """
        return self._k

    @property
    def d(self):
        r"""
        int: The design distance :math:`d` of the :math:`[n, k, d]_2` code. The minimum distance of a BCH code
        may be greater than the design distance, :math:`d_{min} \ge d`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.d
        """
        return 2*self.t + 1

    @property
    def t(self):
        """
        int: The error-correcting capability of the code. The code can correct :math:`t` bit errors in a codeword.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.t
        """
        return self._t

    @property
    def systematic(self):
        """
        bool: Indicates if the code is configured to return codewords in systematic form.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.systematic
        """
        return self._systematic

    @property
    def generator_poly(self):
        """
        galois.Poly: The generator polynomial :math:`g(x)` whose roots are :obj:`roots`.

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
    def roots(self):
        r"""
        galois.FieldArray: The :math:`2t` roots of the generator polynomial. These are consecutive powers of :math:`\alpha`, specifically
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
    def G(self):
        r"""
        galois.GF2: The generator matrix :math:`\mathbf{G}` with shape :math:`(k, n)`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.G
        """
        return self._G

    @property
    def H(self):
        r"""
        galois.FieldArray: The parity-check matrix :math:`\mathbf{H}` with shape :math:`(2t, n)`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.H
        """
        return self._H

    @property
    def is_primitive(self):
        """
        bool: Indicates if the BCH code is primitive, meaning :math:`n = 2^m - 1`.

        Examples
        --------
        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            bch.is_primitive
        """
        return self._is_primitive

    @property
    def is_narrow_sense(self):
        r"""
        bool: Indicates if the BCH code is narrow sense, meaning the roots of the generator polynomial are consecutive
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


###############################################################################
# JIT-compiled implementation of the specified functions
###############################################################################

DECODE_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], int64, int64, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, UNARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, _lfsr.BERLEKAMP_MASSEY_CALCULATE_SIG, POLY_ROOTS_CALCULATE_SIG, int64, int64, int64))

def decode_calculate(codeword, syndrome, t, primitive_element, ADD, SUBTRACT, MULTIPLY, RECIPROCAL, POWER, BERLEKAMP_MASSEY, POLY_ROOTS, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    """
    References
    ----------
    * S. Lin and D. Costello. Error Control Coding. Section 7.4.
    """
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
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
            sigma_rev = BERLEKAMP_MASSEY(syndrome[i,::-1], ADD, SUBTRACT, MULTIPLY, RECIPROCAL, *args)
            v = sigma_rev.size - 1  # The number of errors

            if v > t:
                dec_codeword[i, -1] = -1
                continue

            # Compute βi, the roots of σ(x^-v) which are the inverse roots of σ(x)
            degrees = np.arange(sigma_rev.size - 1, -1, -1)
            results = POLY_ROOTS(degrees, sigma_rev, primitive_element, ADD, MULTIPLY, POWER, *args)
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
