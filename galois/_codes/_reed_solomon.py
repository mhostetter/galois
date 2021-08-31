import numba
from numba import int64
import numpy as np

from .. import _lfsr
from .._factor import factors
from .._factory import Field, matlab_primitive_poly
from .._fields import FieldClass
from .._overrides import set_module
from .._polys import Poly

from ._cyclic import poly_to_generator_matrix, roots_to_parity_check_matrix

__all__ = ["ReedSolomon"]


@set_module("galois")
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
    """
    # pylint: disable=no-member

    def __init__(self, n, k, c=1, primitive_poly=None, primitive_element=None, systematic=True):
        r"""
        Constructs a general :math:`\textrm{RS}(n, k)` code.

        Parameters
        ----------
        n : int
            The codeword size :math:`n`, must be :math:`n = q - 1` where :math:`q` is a prime power.
        k : int
            The message size :math:`k`. The error-correcting capability :math:`t` is defined by :math:`n - k = 2t`.
        c : int, optional
            The first consecutive power of :math:`\alpha`. The default is 1.
        primitive_poly : galois.Poly, optional
            Optionally specify the primitive polynomial that defines the extension field :math:`\mathrm{GF}(q)`. The default is
            `None` which uses Matlab's default, see :func:`galois.matlab_primitive_poly`. Matlab tends to use the lexicographically-minimal
            primitive polynomial as a default instead of the Conway polynomial.
        primitive_element : int, galois.Poly, optional
            Optionally specify the primitive element :math:`\alpha` of :math:`\mathrm{GF}(q)` whose powers are roots of the generator polynomial :math:`g(x)`.
            The default is `None` which uses the lexicographically-minimal primitive element in :math:`\mathrm{GF}(q)`, see
            :func:`galois.primitive_element`.
        systematic : bool, optional
            Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
            appended. The default is `True`.

        Returns
        -------
        galois.ReedSolomon
            A general :math:`\textrm{RS}(n, k)` code object.
        """
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
        if not isinstance(c, (int, np.integer)):
            raise TypeError(f"Argument `c` must be an integer, not {type(c)}.")
        if not isinstance(primitive_poly, (type(None), int, Poly)):
            raise TypeError(f"Argument `primitive_poly` must be an int or galois.Poly, not {type(primitive_poly)}.")
        if not isinstance(systematic, bool):
            raise TypeError(f"Argument `systematic` must be a bool, not {type(systematic)}.")

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
        self._systematic = systematic

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

        # Pre-compile the arithmetic methods
        self._add_jit = self.field._func_calculate("add")
        self._subtract_jit = self.field._func_calculate("subtract")
        self._multiply_jit = self.field._func_calculate("multiply")
        self._reciprocal_jit = self.field._func_calculate("reciprocal")
        self._power_jit = self.field._func_calculate("power")

        # Pre-compile the JIT functions
        self._berlekamp_massey_jit = _lfsr.jit_calculate("berlekamp_massey")
        self._poly_divmod_jit = self.field._function("poly_divmod")
        self._poly_roots_jit = self.field._function("poly_roots")
        self._poly_eval_jit = self.field._function("poly_evaluate")
        self._convolve_jit = self.field._function("convolve")

        # Pre-compile the JIT decoder
        self._decode_jit = numba.jit(DECODE_CALCULATE_SIG.signature, nopython=True, cache=True)(decode_calculate)

    def __str__(self):
        return f"<Reed-Solomon Code: [{self.n}, {self.k}, {self.d}] over {self.field.name}>"

    def __repr__(self):
        return str(self)

    def encode(self, message, parity_only=False):
        r"""
        Encodes the message :math:`\mathbf{m}` into the Reed-Solomon codeword :math:`\mathbf{c}`.

        Parameters
        ----------
        message : numpy.ndarray, galois.FieldArray
            The message as either a :math:`k`-length vector or :math:`(N, k)` matrix, where :math:`N` is the number
            of messages. For systematic codes, message lengths less than :math:`k` may be provided to produce
            shortened codewords.
        parity_only : bool, optional
            Optionally specify whether to return only the parity symbols. This only applies to systematic codes.
            The default is `False`.

        Returns
        -------
        numpy.ndarray, galois.FieldArray
            The codeword as either a :math:`n`-length vector or :math:`(N, n)` matrix. The return type matches the
            message type. If `parity_only=True`, the parity symbols are returned as either a :math:`n - k`-length vector or
            :math:`(N, n-k)` matrix.

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
        Encode a single codeword.

        .. ipython:: python

            rs = galois.ReedSolomon(15, 9)
            GF = rs.field
            m = GF.Random(rs.k); m
            c = rs.encode(m); c
            p = rs.encode(m, parity_only=True); p

        Encode a single, shortened codeword.

        .. ipython:: python

            m = GF.Random(rs.k - 4); m
            c = rs.encode(m); c

        Encode a matrix of codewords.

        .. ipython:: python

            m = GF.Random((5, rs.k)); m
            c = rs.encode(m); c
            p = rs.encode(m, parity_only=True); p
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

        ks = message.shape[-1]  # The number of input message symbols (could be less than self.k for shortened codes)

        if parity_only:
            parity = message.view(self.field) @ self.G[-ks:, self.k:]
            return parity.view(type(message))
        elif self.systematic:
            parity = message.view(self.field) @ self.G[-ks:, self.k:]
            return np.hstack((message, parity)).view(type(message))
        else:
            codeword = message.view(self.field) @ self.G
            return codeword.view(type(message))

    def detect(self, codeword):
        r"""
        Detects if errors are present in the Reed-Solomon codeword :math:`\mathbf{c}`.

        The :math:`[n, k, d]_q` Reed-Solomon code has :math:`d_{min} = d` minimum distance. It can detect up
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

            rs = galois.ReedSolomon(15, 9)
            GF = rs.field
            # The minimum distance of the code
            rs.d
            m = GF.Random(rs.k); m
            c = rs.encode(m); c
            rs.detect(c)

        Detect :math:`d_{min}-1` errors in a received codeword.

        .. ipython:: python

            # Corrupt the first `d - 1` symbols in the codeword
            c[0:rs.d - 1] += GF(13)
            rs.detect(c)
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
        ns = codeword.shape[-1]  # The number of input codeword symbols (could be less than self.n for shortened codes)

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
        Decodes the Reed-Solomon codeword :math:`\mathbf{c}` into the message :math:`\mathbf{m}`.

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
            Optional return argument of the number of corrected symbol errors as either a scalar or :math:`n`-length vector.
            Valid number of corrections are in :math:`[0, t]`. If a codeword has too many errors and cannot be corrected,
            -1 will be returned.

        Notes
        -----
        The codeword vector :math:`\mathbf{c}` is defined as :math:`\mathbf{c} = [c_{n-1}, \dots, c_1, c_0] \in \mathrm{GF}(q)^n`,
        which corresponds to the codeword polynomial :math:`c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0`. The message vector :math:`\mathbf{m}`
        is defined as :math:`\mathbf{m} = [m_{k-1}, \dots, m_1, m_0] \in \mathrm{GF}(q)^k`, which corresponds to the message
        polynomial :math:`m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0`.

        In decoding, the syndrome vector :math:`s` is computed by :math:`\mathbf{s} = \mathbf{c}\mathbf{H}^T`, where
        :math:`\mathbf{H}` is the parity-check matrix. The equivalent polynomial operation is :math:`s(x) = c(x)\ \textrm{mod}\ g(x)`.
        A syndrome of zeros indicates the received codeword is a valid codeword and there are no errors. If the syndrome is non-zero,
        the decoder will find an error-locator polynomial :math:`\sigma(x)` and the corresponding error locations and values.

        For the shortened :math:`\textrm{RS}(n-s, k-s)` code (only applicable for systematic codes), pass :math:`n-s` symbols into
        :func:`decode` to return the :math:`k-s`-symbol message.

        Examples
        --------
        Decode a single codeword.

        .. ipython:: python

            rs = galois.ReedSolomon(15, 9)
            GF = rs.field
            m = GF.Random(rs.k); m
            c = rs.encode(m); c
            # Corrupt the first symbol in the codeword
            c[0] += GF(13)
            dec_m = rs.decode(c); dec_m
            np.array_equal(dec_m, m)

            # Instruct the decoder to return the number of corrected symbol errors
            dec_m, N = rs.decode(c, errors=True); dec_m, N
            np.array_equal(dec_m, m)

        Decode a single, shortened codeword.

        .. ipython:: python

            m = GF.Random(rs.k - 4); m
            c = rs.encode(m); c
            # Corrupt the first symbol in the codeword
            c[0] += GF(13)
            dec_m = rs.decode(c); dec_m
            np.array_equal(dec_m, m)

        Decode a matrix of codewords.

        .. ipython:: python

            m = GF.Random((5, rs.k)); m
            c = rs.encode(m); c
            # Corrupt the first symbol in each codeword
            c[:,0] += GF(13)
            dec_m = rs.decode(c); dec_m
            np.array_equal(dec_m, m)

            # Instruct the decoder to return the number of corrected symbol errors
            dec_m, N = rs.decode(c, errors=True); dec_m, N
            np.array_equal(dec_m, m)
        """
        if not isinstance(codeword, np.ndarray):
            raise TypeError(f"Argument `codeword` must be a subclass of np.ndarray (or a galois.FieldArray), not {type(codeword)}.")
        if self.systematic:
            if not codeword.shape[-1] <= self.n:
                raise ValueError(f"For a systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension less than or equal to {self.n}, not shape {codeword.shape}.")
        else:
            if not codeword.shape[-1] == self.n:
                raise ValueError(f"For a non-systematic code, argument `codeword` must be a 1-D or 2-D array with last dimension equal to {self.n}, not shape {codeword.shape}.")

        codeword_1d = codeword.ndim == 1
        dtype = codeword.dtype
        ns = codeword.shape[-1]  # The number of input codeword symbols (could be less than self.n for shortened codes)
        ks = self.k - (self.n - ns)  # The equivalent number of input message symbols (could be less than self.k for shortened codes)

        # Make codeword 2-D for array processing
        codeword = np.atleast_2d(codeword)

        # Compute the syndrome by matrix multiplying with the parity-check matrix
        syndrome = codeword.view(self.field) @ self.H[:,-ns:].T

        if self.field.ufunc_mode != "python-calculate":
            dec_codeword =  self._decode_jit(codeword.astype(np.int64), syndrome.astype(np.int64), self.t, int(self.field.primitive_element), self._add_jit, self._subtract_jit, self._multiply_jit, self._reciprocal_jit, self._power_jit, self._berlekamp_massey_jit, self._poly_roots_jit, self._poly_eval_jit, self._convolve_jit, self.field.characteristic, self.field.degree, self.field._irreducible_poly_int)
            N_errors = dec_codeword[:, -1]

            if self.systematic:
                message = dec_codeword[:, 0:ks]
            else:
                message, _ = self.field._poly_divmod(dec_codeword[:, 0:self.n].view(self.field), self.generator_poly.coeffs)
            message = message.astype(dtype).view(type(codeword))

        else:
            raise NotImplementedError("Reed-Solomon codes haven't been implemented for extremely large Galois fields.")

        if codeword_1d:
            message, N_errors = message[0,:], N_errors[0]

        if not errors:
            return message
        else:
            return message, N_errors

    @property
    def field(self):
        r"""
        galois.FieldClass: The Galois field :math:`\mathrm{GF}(q)` that defines the Reed-Solomon code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.field
            print(rs.field.properties)
        """
        return self._field

    @property
    def n(self):
        """
        int: The codeword size :math:`n` of the :math:`[n, k, d]_q` code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.n
        """
        return self._n

    @property
    def k(self):
        """
        int: The message size :math:`k` of the :math:`[n, k, d]_q` code.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.k
        """
        return self._k

    @property
    def d(self):
        """
        int: The design distance :math:`d` of the :math:`[n, k, d]_q` code. The minimum distance of a Reed-Solomon code
        is exactly equal to the design distance, :math:`d_{min} = d`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.d
        """
        return 2*self.t + 1

    @property
    def t(self):
        """
        int: The error-correcting capability of the code. The code can correct :math:`t` symbol errors in a codeword.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.t
        """
        return self._t

    @property
    def systematic(self):
        """
        bool: Indicates if the code is configured to return codewords in systematic form.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.systematic
        """
        return self._systematic

    @property
    def generator_poly(self):
        """
        galois.Poly: The generator polynomial :math:`g(x)` whose roots are :obj:`roots`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.generator_poly
            # Evaluate the generator polynomial at its roots
            rs.generator_poly(rs.roots)
        """
        return self._generator_poly

    @property
    def roots(self):
        r"""
        galois.FieldArray: The :math:`2t` roots of the generator polynomial. These are consecutive powers of :math:`\alpha`, specifically
        :math:`\alpha^c, \alpha^{c+1}, \dots, \alpha^{c+2t-1}`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.roots
            # Evaluate the generator polynomial at its roots
            rs.generator_poly(rs.roots)
        """
        return self._roots

    @property
    def c(self):
        """
        int: The degree of the first consecutive root.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.c
        """
        return self._c

    @property
    def G(self):
        r"""
        galois.FieldArray: The generator matrix :math:`\mathbf{G}` with shape :math:`(k, n)`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.G
        """
        return self._G

    @property
    def H(self):
        r"""
        galois.FieldArray: The parity-check matrix :math:`\mathbf{H}` with shape :math:`(2t, n)`.

        Examples
        --------
        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            rs.H
        """
        return self._H

    @property
    def is_narrow_sense(self):
        r"""
        bool: Indicates if the Reed-Solomon code is narrow sense, meaning the roots of the generator polynomial are consecutive
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


###############################################################################
# JIT-compiled implementation of the specified functions
###############################################################################

DECODE_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], int64, int64, FieldClass._BINARY_CALCULATE_SIG, FieldClass._BINARY_CALCULATE_SIG, FieldClass._BINARY_CALCULATE_SIG, FieldClass._UNARY_CALCULATE_SIG, FieldClass._BINARY_CALCULATE_SIG, _lfsr.BERLEKAMP_MASSEY_CALCULATE_SIG, FieldClass._POLY_ROOTS_CALCULATE_SIG, FieldClass._POLY_EVALUATE_CALCULATE_SIG, FieldClass._CONVOLVE_CALCULATE_SIG, int64, int64, int64))

def decode_calculate(codeword, syndrome, t, primitive_element, ADD, SUBTRACT, MULTIPLY, RECIPROCAL, POWER, BERLEKAMP_MASSEY, POLY_ROOTS, POLY_EVAL, CONVOLVE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
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

            # Compute the error-locator polynomial σ(x) its v-reversal σ(x^-v), since the syndrome is passed in backwards
            sigma = BERLEKAMP_MASSEY(syndrome[i,:], ADD, SUBTRACT, MULTIPLY, RECIPROCAL, *args)
            sigma_rev = BERLEKAMP_MASSEY(syndrome[i,::-1], ADD, SUBTRACT, MULTIPLY, RECIPROCAL, *args)
            v = sigma.size - 1  # The number of errors

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

            # Compute σ'(x)
            sigma_prime = np.zeros(v, dtype=np.int64)
            for j in range(v):
                degree = v - j
                sigma_prime[j] = MULTIPLY(degree % CHARACTERISTIC, sigma[j], *args)  # Scalar multiplication

            # The error-value evalulator polynomial Z0(x) = S0*σ0 + (S1*σ0 + S0*σ1)*x + (S2*σ0 + S1*σ1 + S0*σ2)*x^2 + ...
            # with degree v-1
            Z0 = CONVOLVE(sigma[-v:], syndrome[i,0:v][::-1], ADD, MULTIPLY, *args)[-v:]

            # The error value δi = -Z0(βi^-1) / σ'(βi^-1)
            for j in range(v):
                beta_inv = RECIPROCAL(beta[j], *args)
                Z0_i = POLY_EVAL(Z0, np.array([beta_inv], dtype=dtype), ADD, MULTIPLY, *args)[0]  # NOTE: poly_eval() expects a 1-D array of values
                sigma_prime_i = POLY_EVAL(sigma_prime, np.array([beta_inv], dtype=dtype), ADD, MULTIPLY, *args)[0]  # NOTE: poly_eval() expects a 1-D array of values
                delta_i = MULTIPLY(SUBTRACT(0, Z0_i, *args), RECIPROCAL(sigma_prime_i, *args), *args)
                dec_codeword[i, n - 1 - error_locations[j]] = SUBTRACT(dec_codeword[i, n - 1 - error_locations[j]], delta_i, *args)
            dec_codeword[i, -1] = v  # The number of corrected errors

    return dec_codeword
