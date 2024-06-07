"""
A module containing general Bose-Chaudhuri-Hocquenghem (BCH) codes over GF(q).
"""

from __future__ import annotations

from typing import Type, overload

import numba
import numpy as np
from numba import int64
from typing_extensions import Literal

from .._domains._function import Function
from .._fields import GF2, Field, FieldArray
from .._helper import export, extend_docstring, verify_isinstance, verify_issubclass
from .._lfsr import berlekamp_massey_jit
from .._math import ilog
from .._polys import Poly, matlab_primitive_poly
from .._polys._dense import evaluate_elementwise_jit, roots_jit
from ..typing import ArrayLike, ElementLike
from ._cyclic import _CyclicCode


@export
class BCH(_CyclicCode):
    r"""
    A general $\textrm{BCH}(n, k)$ code over $\mathrm{GF}(q)$.

    A $\textrm{BCH}(n, k)$ code is a $[n, k, d]_q$ linear block code with codeword size $n$, message
    size $k$, minimum distance $d$, and symbols taken from an alphabet of size $q$.

    .. info::
        :title: Shortened codes

        To create the shortened $\textrm{BCH}(n-s, k-s)$ code, construct the full-sized
        $\textrm{BCH}(n, k)$ code and then pass $k-s$ symbols into :func:`encode` and $n-s$ symbols
        into :func:`decode()`. Shortened codes are only applicable for systematic codes.

    A BCH code is a cyclic code over $\mathrm{GF}(q)$ with generator polynomial $g(x)$. The generator
    polynomial is over $\mathrm{GF}(q)$ and has $d-1$ roots $\alpha^c, \dots, \alpha^{c+d-2}$ when
    evaluated in $\mathrm{GF}(q^m)$. The element $\alpha$ is a primitive $n$-th root of unity in
    $\mathrm{GF}(q^m)$.

    $$g(x) = \textrm{LCM}(m_{\alpha^c}(x), \dots, m_{\alpha^{c+d-2}}(x))$$

    Examples:
        Construct a binary $\textrm{BCH}(15, 7)$ code.

        .. ipython:: python

            bch = galois.BCH(15, 7); bch
            GF = bch.field; GF

        Encode a message.

        .. ipython:: python

            m = GF.Random(bch.k); m
            c = bch.encode(m); c

        Corrupt the codeword and decode the message.

        .. ipython:: python

            # Corrupt the first symbol in the codeword
            c[0] ^= 1; c
            dec_m = bch.decode(c); dec_m
            np.array_equal(dec_m, m)

        Instruct the decoder to return the number of corrected symbol errors.

        .. ipython:: python

            dec_m, N = bch.decode(c, errors=True); dec_m, N
            np.array_equal(dec_m, m)

    Group:
        fec
    """

    def __init__(
        self,
        n: int,
        k: int | None = None,
        d: int | None = None,
        field: Type[FieldArray] | None = None,
        extension_field: Type[FieldArray] | None = None,
        alpha: ElementLike | None = None,
        c: int = 1,
        systematic: bool = True,
    ):
        r"""
        Constructs a general $\textrm{BCH}(n, k)$ code over $\mathrm{GF}(q)$.

        Arguments:
            n: The codeword size $n$. If $n = q^m - 1$, the BCH code is *primitive*.
            k: The message size $k$.

                .. important::
                    Either `k` or `d` must be provided to define the code. Both may be provided as long as they are
                    consistent.

            d: The design distance $d$. This defines the number of roots $d - 1$ in the generator
                polynomial $g(x)$ over $\mathrm{GF}(q^m)$.
            field: The Galois field $\mathrm{GF}(q)$ that defines the alphabet of the codeword symbols.
                The default is `None` which corresponds to $\mathrm{GF}(2)$.
            extension_field: The Galois field $\mathrm{GF}(q^m)$ that defines the syndrome arithmetic.
                The default is `None` which corresponds to $\mathrm{GF}(q^m)$ where
                $q^{m - 1} \le n < q^m$. The default extension field will use `matlab_primitive_poly(q, m)`
                for the irreducible polynomial.
            alpha: A primitive $n$-th root of unity $\alpha$ in $\mathrm{GF}(q^m)$ that defines the
                $\alpha^c, \dots, \alpha^{c+d-2}$ roots of the generator polynomial $g(x)$.
            c: The first consecutive power $c$ of $\alpha$ that defines the
                $\alpha^c, \dots, \alpha^{c+d-2}$ roots of the generator polynomial $g(x)$.
                The default is 1. If $c = 1$, the BCH code is *narrow-sense*.
            systematic: Indicates if the encoding should be systematic, meaning the codeword is the message with
                parity appended. The default is `True`.

        See Also:
            matlab_primitive_poly, FieldArray.primitive_root_of_unity

        Examples:
            Construct a binary primitive, narrow-sense $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                galois.BCH(15, 7)
                galois.BCH(15, d=5)
                galois.BCH(15, 7, 5)

            Construct a primitive, narrow-sense $\textrm{BCH}(26, 17)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                GF = galois.GF(3)
                galois.BCH(26, 17, field=GF)
                galois.BCH(26, d=5, field=GF)
                galois.BCH(26, 17, 5, field=GF)

            Construct a non-primitive, narrow-sense $\textrm{BCH}(13, 4)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                GF = galois.GF(3)
                galois.BCH(13, 4, field=GF)
                galois.BCH(13, d=7, field=GF)
                galois.BCH(13, 4, 7, field=GF)

            Discover primitive BCH codes over $\mathrm{GF}(5)$ by looping over the design distance.

            .. ipython:: python

                GF = galois.GF(5)
                n = 5**2 - 1
                for d in range(2, 11):
                    bch = galois.BCH(n, d=d, field=GF)
                    print(repr(bch))
        """
        verify_isinstance(n, int)
        verify_isinstance(k, int, optional=True)
        verify_isinstance(d, int, optional=True)
        verify_issubclass(field, FieldArray, optional=True)
        verify_issubclass(extension_field, FieldArray, optional=True)
        verify_isinstance(c, int)

        if d is not None and not d >= 1:
            raise ValueError(f"Argument 'd' must be at least 1, not {d}.")
        if not c >= 0:
            raise ValueError(f"Argument 'c' must be at least 0, not {c}.")

        if field is None:
            field = GF2
        if not field.is_prime_field:
            raise ValueError(
                "Current BCH codes over GF(q) for prime power q are not supported. "
                "Proper Galois field towers are needed first."
            )
        q = field.order  # The size of the codeword alphabet

        if extension_field is None:
            m = ilog(n, q) + 1
            assert q ** (m - 1) < n + 1 <= q**m
            irreducible_poly = matlab_primitive_poly(q, m)
            extension_field = Field(q**m, irreducible_poly=irreducible_poly)

        if alpha is None:
            alpha = extension_field.primitive_root_of_unity(n)
        else:
            alpha = extension_field(alpha)

        if d is not None:
            generator_poly, roots = _generator_poly_from_d(d, field, alpha, c)
            # Check if both `k` and `d` were provided that the code is consistent
            kk = n - generator_poly.degree
            if not k in [None, kk]:
                raise ValueError(
                    f"The requested [{n}, {k}, {d}] code is not consistent. "
                    f"When designing the code with design distance {d}, the resulting code is [{n}, {kk}, {d}]."
                )
            k = kk
        elif k is not None:
            generator_poly, roots = _generator_poly_from_k(n, k, field, extension_field, alpha, c)
            # We know `d` wasn't provided, otherwise the previous `if` would have executed
            d = roots.size + 1
        else:
            raise ValueError("Argument 'k' or 'd' must be provided to define the code size.")

        # Set BCH specific attributes
        self._extension_field = extension_field
        self._alpha = alpha
        self._c = c
        self._is_primitive = n == extension_field.order - 1
        self._is_narrow_sense = c == 1

        super().__init__(n, k, d, generator_poly, roots, systematic)

    def __repr__(self) -> str:
        r"""
        A terse representation of the BCH code.

        Examples:
            Construct a binary primitive $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7)
                bch

            Construct a primitive $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3))
                bch

            Construct a non-primitive $\textrm{BCH}(13, 4)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3))
                bch
        """
        return f"<BCH Code: [{self.n}, {self.k}, {self.d}] over {self.field.name}>"

    def __str__(self) -> str:
        r"""
        A formatted string with relevant properties of the BCH code.

        Examples:
            Construct a binary primitive $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7)
                print(bch)

            Construct a primitive $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3))
                print(bch)

            Construct a non-primitive $\textrm{BCH}(13, 4)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3))
                print(bch)
        """
        string = "BCH Code:"
        string += f"\n  [n, k, d]: [{self.n}, {self.k}, {self.d}]"
        string += f"\n  field: {self.field.name}"
        string += f"\n  extension_field: {self.extension_field.name}"
        string += f"\n  generator_poly: {self.generator_poly}"
        string += f"\n  is_primitive: {self.is_primitive}"
        string += f"\n  is_narrow_sense: {self.is_narrow_sense}"
        string += f"\n  is_systematic: {self.is_systematic}"

        return string

    @extend_docstring(
        _CyclicCode.encode,
        {},
        r"""
        Examples:
            .. md-tab-set::

                .. md-tab-item:: Vector

                    Encode a single message using the $\textrm{BCH}(15, 7)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random(bch.k); m
                        c = bch.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = bch.encode(m, output="parity"); p

                .. md-tab-item:: Vector (shortened)

                    Encode a single message using the shortened $\textrm{BCH}(12, 4)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random(bch.k - 3); m
                        c = bch.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = bch.encode(m, output="parity"); p

                .. md-tab-item:: Matrix

                    Encode a matrix of three messages using the $\textrm{BCH}(15, 7)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random((3, bch.k)); m
                        c = bch.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = bch.encode(m, output="parity"); p

                .. md-tab-item:: Matrix (shortened)

                    Encode a matrix of three messages using the shortened $\textrm{BCH}(12, 4)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random((3, bch.k - 3)); m
                        c = bch.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = bch.encode(m, output="parity"); p
        """,
    )
    def encode(self, message: ArrayLike, output: Literal["codeword", "parity"] = "codeword") -> FieldArray:
        return super().encode(message, output=output)

    @extend_docstring(
        _CyclicCode.detect,
        {},
        r"""
        Examples:
            .. md-tab-set::

                .. md-tab-item:: Vector

                    Encode a single message using the $\textrm{BCH}(15, 7)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random(bch.k); m
                        c = bch.encode(m); c

                    Detect no errors in the valid codeword.

                    .. ipython:: python

                        bch.detect(c)

                    Detect $d_{min}-1$ errors in the codeword.

                    .. ipython:: python

                        bch.d
                        c[0:bch.d - 1] ^= 1; c
                        bch.detect(c)

                .. md-tab-item:: Vector (shortened)

                    Encode a single message using the shortened $\textrm{BCH}(12, 4)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random(bch.k - 3); m
                        c = bch.encode(m); c

                    Detect no errors in the valid codeword.

                    .. ipython:: python

                        bch.detect(c)

                    Detect $d_{min}-1$ errors in the codeword.

                    .. ipython:: python

                        bch.d
                        c[0:bch.d - 1] ^= 1; c
                        bch.detect(c)

                .. md-tab-item:: Matrix

                    Encode a matrix of three messages using the $\textrm{BCH}(15, 7)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random((3, bch.k)); m
                        c = bch.encode(m); c

                    Detect no errors in the valid codewords.

                    .. ipython:: python

                        bch.detect(c)

                    Detect one, two, and $d_{min}-1$ errors in the codewords.

                    .. ipython:: python

                        bch.d
                        c[0, 0:1] ^= 1
                        c[1, 0:2] ^= 1
                        c[2, 0:bch.d - 1] ^= 1
                        c
                        bch.detect(c)

                .. md-tab-item:: Matrix (shortened)

                    Encode a matrix of three messages using the shortened $\textrm{BCH}(12, 4)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random((3, bch.k - 3)); m
                        c = bch.encode(m); c

                    Detect no errors in the valid codewords.

                    .. ipython:: python

                        bch.detect(c)

                    Detect one, two, and $d_{min}-1$ errors in the codewords.

                    .. ipython:: python

                        bch.d
                        c[0, 0:1] ^= 1
                        c[1, 0:2] ^= 1
                        c[2, 0:bch.d - 1] ^= 1
                        c
                        bch.detect(c)
        """,
    )
    def detect(self, codeword: ArrayLike) -> bool | np.ndarray:
        return super().detect(codeword)

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[False] = False,
    ) -> FieldArray: ...

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[True] = True,
    ) -> tuple[FieldArray, int | np.ndarray]: ...

    @extend_docstring(
        _CyclicCode.decode,
        {},
        r"""
        In decoding, the syndrome vector $\mathbf{s}$ is computed by evaluating the received codeword
        $\mathbf{r}$ in the extension field $\mathrm{GF}(q^m)$ at the roots
        $\alpha^c, \dots, \alpha^{c+d-2}$ of the generator polynomial $g(x)$. The equivalent polynomial
        operation computes the remainder of $r(x)$ by $g(x)$ in the extension field
        $\mathrm{GF}(q^m)$.

        $$\mathbf{s} = [r(\alpha^c),\ \dots,\ r(\alpha^{c+d-2})] \in \mathrm{GF}(q^m)^{d-1}$$

        $$s(x) = r(x)\ \textrm{mod}\ g(x) \in \mathrm{GF}(q^m)[x]$$

        A syndrome of zeros indicates the received codeword is a valid codeword and there are no errors. If the
        syndrome is non-zero, the decoder will find an error-locator polynomial $\sigma(x)$ and the corresponding
        error locations and values.

        Note:
            The $[n, k, d]_q$ code has $d_{min} \ge d$ minimum distance. It can detect up
            to $d_{min}-1$ errors.

        Examples:
            .. md-tab-set::

                .. md-tab-item:: Vector

                    Encode a single message using the $\textrm{BCH}(15, 7)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random(bch.k); m
                        c = bch.encode(m); c

                    Corrupt $t$ symbols of the codeword.

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

                .. md-tab-item:: Vector (shortened)

                    Encode a single message using the shortened $\textrm{BCH}(12, 4)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random(bch.k - 3); m
                        c = bch.encode(m); c

                    Corrupt $t$ symbols of the codeword.

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

                .. md-tab-item:: Matrix

                    Encode a matrix of three messages using the $\textrm{BCH}(15, 7)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random((3, bch.k)); m
                        c = bch.encode(m); c

                    Corrupt the codeword. Add zero errors to the first codeword, one to the second, and two to the
                    third.

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

                .. md-tab-item:: Matrix (shortened)

                    Encode a matrix of three messages using the shortened $\textrm{BCH}(12, 4)$ code.

                    .. ipython:: python

                        bch = galois.BCH(15, 7)
                        GF = bch.field
                        m = GF.Random((3, bch.k - 3)); m
                        c = bch.encode(m); c

                    Corrupt the codeword. Add zero errors to the first codeword, one to the second, and two to the
                    third.

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
        """,
    )
    def decode(self, codeword, output="message", errors=False):
        return super().decode(codeword, output=output, errors=errors)

    def _decode_codeword(self, codeword: FieldArray) -> tuple[FieldArray, np.ndarray]:
        func = bch_decode_jit(self.field, self.extension_field)
        dec_codeword, N_errors = func(codeword, self.n, int(self.alpha), self.c, self.roots)
        dec_codeword = dec_codeword.view(self.field)
        return dec_codeword, N_errors

    @property
    @extend_docstring(
        _CyclicCode.field,
        {},
        r"""
        Examples:
            Construct a binary $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.field
                print(bch.field.properties)

            Construct a $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3)); bch
                bch.field
                print(bch.field.properties)
        """,
    )
    def field(self) -> Type[FieldArray]:
        return super().field

    @property
    def extension_field(self) -> Type[FieldArray]:
        r"""
        The Galois field $\mathrm{GF}(q^m)$ that defines the BCH syndrome arithmetic.

        Examples:
            Construct a binary $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.extension_field
                print(bch.extension_field.properties)

            Construct a $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3)); bch
                bch.extension_field
                print(bch.extension_field.properties)
        """
        return self._extension_field

    @extend_docstring(
        _CyclicCode.n,
        {},
        r"""
        Examples:
            Construct a binary $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.n

            Construct a $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3)); bch
                bch.n
        """,
    )
    @property
    def n(self) -> int:
        return super().n

    @extend_docstring(
        _CyclicCode.k,
        {},
        r"""
        Examples:
            Construct a binary $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.k

            Construct a $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3)); bch
                bch.k
        """,
    )
    @property
    def k(self) -> int:
        return super().k

    @extend_docstring(
        _CyclicCode.d,
        {},
        r"""
        Notes:
            The minimum distance of a BCH code may be greater than the design distance, i.e. $d_{min} \ge d$.

        Examples:
            Construct a binary $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.d

            Construct a $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3)); bch
                bch.d
        """,
    )
    @property
    def d(self) -> int:
        return super().d

    @extend_docstring(
        _CyclicCode.t,
        {},
        r"""
        Examples:
            Construct a binary $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.t

            Construct a $\textrm{BCH}(26, 14)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(26, 14, field=galois.GF(3)); bch
                bch.t
        """,
    )
    @property
    def t(self) -> int:
        return super().t

    @extend_docstring(
        _CyclicCode.generator_poly,
        {},
        r"""
        Examples:
            Construct a binary narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha$.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.generator_poly
                bch.roots
                # Evaluate the generator polynomial at its roots in GF(q^m)
                bch.generator_poly(bch.roots, field=bch.extension_field)

            Construct a binary non-narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha^3$. Notice the design distance of this code is only 3 and it only has 2 roots
            in $\mathrm{GF}(2^4)$.

            .. ipython:: python

                bch = galois.BCH(15, 7, c=3); bch
                bch.generator_poly
                bch.roots
                # Evaluate the generator polynomial at its roots in GF(q^m)
                bch.generator_poly(bch.roots, field=bch.extension_field)
        """,
    )
    @property
    def generator_poly(self) -> Poly:
        return super().generator_poly

    @extend_docstring(
        _CyclicCode.parity_check_poly,
        {},
        r"""
        Examples:
            Construct a binary primitive $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.parity_check_poly
                bch.H

            Construct a non-primitive $\textrm{BCH}(13, 4)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3)); bch
                bch.parity_check_poly
                bch.H
        """,
    )
    @property
    def parity_check_poly(self) -> Poly:
        return super().parity_check_poly

    @extend_docstring(
        _CyclicCode.roots,
        {},
        r"""
        These are consecutive powers of $\alpha^c$, specifically $\alpha^c, \dots, \alpha^{c+d-2}$.

        Examples:
            Construct a binary narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha$.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.roots
                bch.generator_poly
                # Evaluate the generator polynomial at its roots in GF(q^m)
                bch.generator_poly(bch.roots, field=bch.extension_field)

            Construct a binary non-narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha^3$. Notice the design distance of this code is only 3 and it only has 2 roots
            in $\mathrm{GF}(2^4)$.

            .. ipython:: python

                bch = galois.BCH(15, 7, c=3); bch
                bch.roots
                bch.generator_poly
                # Evaluate the generator polynomial at its roots in GF(q^m)
                bch.generator_poly(bch.roots, field=bch.extension_field)
        """,
    )
    @property
    def roots(self) -> FieldArray:
        return super().roots

    @property
    def alpha(self) -> FieldArray:
        r"""
        A primitive $n$-th root of unity $\alpha$ in $\mathrm{GF}(q^m)$ whose consecutive powers
        $\alpha^c, \dots, \alpha^{c+d-2}$ are roots of the generator polynomial $g(x)$
        in $\mathrm{GF}(q^m)$.

        Examples:
            Construct a binary primitive $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.alpha
                bch.roots[0] == bch.alpha ** bch.c
                bch.alpha.multiplicative_order() == bch.n

            Construct a non-primitive $\textrm{BCH}(13, 7)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 7, field=galois.GF(3)); bch
                bch.alpha
                bch.roots[0] == bch.alpha ** bch.c
                bch.alpha.multiplicative_order() == bch.n

        Group:
            Polynomials

        Order:
            72
        """
        return self._alpha

    @property
    def c(self) -> int:
        r"""
        The first consecutive power $c$ of $\alpha$ that defines the roots
        $\alpha^c, \dots, \alpha^{c+d-2}$ of the generator polynomial $g(x)$.

        Examples:
            Construct a binary narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha$.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.c
                bch.roots[0] == bch.alpha ** bch.c

            Construct a binary non-narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha^3$. Notice the design distance of this code is only 3.

            .. ipython:: python

                bch = galois.BCH(15, 7, c=3); bch
                bch.c
                bch.roots[0] == bch.alpha ** bch.c

        Group:
            Polynomials

        Order:
            72
        """
        return self._c

    @extend_docstring(
        _CyclicCode.G,
        {},
        r"""
        Examples:
            Construct a binary primitive $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.G

            Construct a non-primitive $\textrm{BCH}(13, 4)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3)); bch
                bch.G

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3), systematic=False); bch
                bch.G
                bch.generator_poly
        """,
    )
    @property
    def G(self) -> FieldArray:
        return super().G

    @extend_docstring(
        _CyclicCode.H,
        {},
        r"""
        Examples:
            Construct a binary primitive $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.H
                bch.parity_check_poly

            Construct a non-primitive $\textrm{BCH}(13, 4)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3)); bch
                bch.H
                bch.parity_check_poly
        """,
    )
    @property
    def H(self) -> FieldArray:
        return super().H

    @property
    def is_primitive(self) -> bool:
        r"""
        Indicates if the BCH code is *primitive*, meaning $n = q^m - 1$.

        Examples:
            Construct a binary primitive $\textrm{BCH}(15, 7)$ code.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.is_primitive
                bch.n == bch.extension_field.order - 1

            Construct a non-primitive $\textrm{BCH}(13, 7)$ code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 7, field=galois.GF(3)); bch
                bch.is_primitive
                bch.n == bch.extension_field.order - 1
        """
        return self._is_primitive

    @property
    def is_narrow_sense(self) -> bool:
        r"""
        Indicates if the BCH code is *narrow-sense*, meaning the roots of the generator polynomial are consecutive
        powers of $\alpha$ starting at 1, that is $\alpha, \dots, \alpha^{d-1}$.

        Examples:
            Construct a binary narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha$.

            .. ipython:: python

                bch = galois.BCH(15, 7); bch
                bch.is_narrow_sense
                bch.c == 1
                bch.generator_poly
                bch.roots

            Construct a binary non-narrow-sense $\textrm{BCH}(15, 7)$ code with first consecutive root
            $\alpha^3$. Notice the design distance of this code is only 3.

            .. ipython:: python

                bch = galois.BCH(15, 7, c=3); bch
                bch.is_narrow_sense
                bch.c == 1
                bch.generator_poly
                bch.roots
        """
        return self._is_narrow_sense

    @extend_docstring(
        _CyclicCode.is_systematic,
        {},
        r"""
        Examples:
            Construct a non-primitive $\textrm{BCH}(13, 4)$ systematic code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3)); bch
                bch.is_systematic
                bch.G

            Construct a non-primitive $\textrm{BCH}(13, 4)$ non-systematic code over $\mathrm{GF}(3)$.

            .. ipython:: python

                bch = galois.BCH(13, 4, field=galois.GF(3), systematic=False); bch
                bch.is_systematic
                bch.G
                bch.generator_poly
        """,
    )
    @property
    def is_systematic(self) -> bool:
        return super().is_systematic


def _generator_poly_from_d(
    d: int,
    field: Type[FieldArray],
    alpha: FieldArray,
    c: int,
) -> tuple[Poly, FieldArray]:
    """
    Determines the BCH generator polynomial from the design distance d.
    """
    generator_poly = Poly.One(field)
    roots = alpha ** (c + np.arange(0, d - 1))

    minimal_polys = []
    for root in roots:
        mi = root.minimal_poly()
        if mi not in minimal_polys:
            generator_poly *= mi
            minimal_polys.append(mi)

    return generator_poly, roots


def _generator_poly_from_k(
    n: int,
    k: int,
    field: Type[FieldArray],
    extension_field: Type[FieldArray],
    alpha: FieldArray,
    c: int,
) -> tuple[Poly, FieldArray]:
    """
    Determines the BCH generator polynomial from the message size k.
    """
    m = ilog(extension_field.order, field.order)

    min_d = (n - k) // m + 1
    max_d = (n - k) + 1
    possible_d = list(range(min_d, max_d + 1))

    # Binary search for a d that creates the BCH(n, k) code
    while len(possible_d) > 0:
        idx = len(possible_d) // 2
        d = possible_d[idx]
        generator_poly, roots = _generator_poly_from_d(d, field, alpha, c)

        if generator_poly.degree < n - k:
            # This d is too small to produce the BCH code
            possible_d = possible_d[idx + 1 :]
        elif generator_poly.degree == n - k:
            # This d produces the correct BCH code size and g(x) is its generator. However, there may also be a
            # larger d that generates a BCH code of the same size, so keep looking.
            break
        else:
            # This d is too large to produce the BCH code
            possible_d = possible_d[0:idx]
    else:
        raise ValueError(f"The BCH({n}, {k}) code over {field.name} with alpha={alpha} and c={c} does not exist.")

    # Single step d to ensure there are no codes of the same size with larger design distance
    # TODO: Can this be another binary search?
    best_generator_poly = generator_poly
    best_roots = roots
    while True:
        d += 1
        generator_poly, roots = _generator_poly_from_d(d, field, alpha, c)

        if generator_poly.degree == n - k:
            # This larger d still produces the same size code
            best_generator_poly = generator_poly
            best_roots = roots
        elif generator_poly.degree > n - k:
            # This d does not produce a valid code, but the previous d (which is "best") did, so use it
            break

    return best_generator_poly, best_roots


class bch_decode_jit(Function):
    """
    Performs general BCH and Reed-Solomon decoding.

    References:
        - Lin, S. and Costello, D. Error Control Coding. Section 7.4.
    """

    def __init__(self, field: Type[FieldArray], extension_field: Type[FieldArray]):
        super().__init__(field)
        self.extension_field = extension_field

    @property
    def key_1(self):
        # Make the key in the cache lookup table specific to both the base field and extension field
        return (
            self.field.characteristic,
            self.field.degree,
            int(self.field.irreducible_poly),
            self.extension_field.characteristic,
            self.extension_field.degree,
            int(self.extension_field.irreducible_poly),
        )

    def __call__(self, codeword, design_n, alpha, c, roots):
        if self.extension_field.ufunc_mode != "python-calculate":
            output = self.jit(codeword.astype(np.int64), design_n, alpha, c, roots.astype(np.int64))
        else:
            output = self.python(codeword.view(np.ndarray), design_n, alpha, c, roots.view(np.ndarray))

        dec_codeword, N_errors = output[:, 0:-1], output[:, -1]
        dec_codeword = dec_codeword.astype(codeword.dtype)
        dec_codeword = dec_codeword.view(self.field)

        return dec_codeword, N_errors

    def set_globals(self):
        global CHARACTERISTIC, SUBTRACT, MULTIPLY, RECIPROCAL, POWER
        global CONVOLVE, POLY_ROOTS, POLY_EVALUATE, BERLEKAMP_MASSEY

        SUBTRACT = self.field._subtract.ufunc_call_only

        CHARACTERISTIC = self.extension_field.characteristic
        MULTIPLY = self.extension_field._multiply.ufunc_call_only
        RECIPROCAL = self.extension_field._reciprocal.ufunc_call_only
        POWER = self.extension_field._power.ufunc_call_only
        CONVOLVE = self.extension_field._convolve.function
        POLY_ROOTS = roots_jit(self.extension_field).function
        POLY_EVALUATE = evaluate_elementwise_jit(self.extension_field).function
        BERLEKAMP_MASSEY = berlekamp_massey_jit(self.extension_field).function

    _SIGNATURE = numba.types.FunctionType(int64[:, :](int64[:, :], int64, int64, int64, int64[:]))

    @staticmethod
    def implementation(codewords, design_n, alpha, c, roots):  # pragma: no cover
        dtype = codewords.dtype
        N = codewords.shape[0]  # The number of codewords
        n = codewords.shape[1]  # The codeword size (could be less than the design n for shortened codes)
        d = roots.size + 1
        t = (d - 1) // 2

        # The last column of the returned decoded codeword is the number of corrected errors
        dec_codewords = np.zeros((N, n + 1), dtype=dtype)
        dec_codewords[:, 0:n] = codewords[:, :]

        for i in range(N):
            # Compute the syndrome by evaluating each codeword at the roots of the generator polynomial.
            # The syndrome vector is S = [S0, S1, ..., S2t-1]
            syndrome = POLY_EVALUATE(codewords[i, :], roots)

            if np.all(syndrome == 0):
                continue

            # The error pattern is defined as the polynomial e(x) = e_j1*x^j1 + e_j2*x^j2 + ... for j1 to jv,
            # implying there are v errors. And δi = e_ji is the i-th error value and βi = α^ji is the i-th
            # error-locator value and ji is the error location.

            # The error-locator polynomial σ(x) = (1 - β1*x)(1 - β2*x)...(1 - βv*x) where βi are the inverse of the
            # roots of σ(x).

            # Compute the error-locator polynomial σ(x)
            # TODO: Re-evaluate these equations since changing BMA to return the characteristic polynomial,
            #       not the feedback polynomial
            sigma = BERLEKAMP_MASSEY(syndrome)[::-1]
            v = sigma.size - 1  # The number of errors, which is the degree of the error-locator polynomial

            if v > t:
                dec_codewords[i, -1] = -1
                continue

            # Compute βi^-1, the roots of σ(x)
            degrees = np.arange(sigma.size - 1, -1, -1)
            results = POLY_ROOTS(degrees, sigma, alpha)
            beta_inv = results[0, :]  # The roots βi^-1 of σ(x)
            error_locations_inv = results[1, :]  # The roots βi^-1 as powers of the primitive element α
            error_locations = -error_locations_inv % design_n  # The error locations as degrees of c(x)

            if np.any(error_locations > n - 1):
                # Indicates there are "errors" in the zero-ed portion of a shortened code, which indicates there are
                # actually more errors than alleged. Return failure to decode.
                dec_codewords[i, -1] = -1
                continue

            if beta_inv.size != v:
                dec_codewords[i, -1] = -1
                continue

            # Compute σ'(x)
            sigma_prime = np.zeros(v, dtype=dtype)
            for j in range(v):
                degree = v - j
                sigma_prime[j] = MULTIPLY(degree % CHARACTERISTIC, sigma[j])  # Scalar multiplication

            # The error-value evaluator polynomial Z0(x) = S0*σ0 + (S1*σ0 + S0*σ1)*x + (S2*σ0 + S1*σ1 + S0*σ2)*x^2 + ...
            # with degree v-1
            Z0 = CONVOLVE(sigma[-v:], syndrome[0:v][::-1])[-v:]

            # The error value δi = -1 * βi^(1-c) * Z0(βi^-1) / σ'(βi^-1)
            for j in range(v):
                beta_i = POWER(beta_inv[j], c - 1)
                # NOTE: poly_eval() expects a 1-D array of values
                Z0_i = POLY_EVALUATE(Z0, np.array([beta_inv[j]], dtype=dtype))[0]
                # NOTE: poly_eval() expects a 1-D array of values
                sigma_prime_i = POLY_EVALUATE(sigma_prime, np.array([beta_inv[j]], dtype=dtype))[0]
                delta_i = MULTIPLY(beta_i, Z0_i)
                delta_i = MULTIPLY(delta_i, RECIPROCAL(sigma_prime_i))
                delta_i = SUBTRACT(0, delta_i)
                dec_codewords[i, n - 1 - error_locations[j]] = SUBTRACT(
                    dec_codewords[i, n - 1 - error_locations[j]], delta_i
                )

            dec_codewords[i, -1] = v  # The number of corrected errors

        return dec_codewords
