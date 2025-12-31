"""
A module containing general Reed-Solomon (RS) codes.
"""

from __future__ import annotations

from typing import Type, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._fields import Field, FieldArray
from .._helper import export, extend_docstring, verify_isinstance, verify_issubclass
from .._math import ilog
from .._polys import Poly, matlab_primitive_poly
from ..typing import ArrayLike, ElementLike
from ._bch import bch_decode_jit
from ._cyclic import _CyclicCode


@export
class ReedSolomon(_CyclicCode):
    r"""
    A general $\mathrm{RS}(n, k)$ code over $\mathrm{GF}(q)$.

    A $\mathrm{RS}(n, k)$ code is a maximum-distance separable (MDS) $[n, k, n - k + 1]_q$ linear block code of
    length $n$, dimension $k$, and minimum distance $d = n - k + 1$, with symbols drawn from $\mathrm{GF}(q)$.
    Reed-Solomon codes achieve the largest possible minimum distance for any $q$-ary linear block code with the
    same length and dimension.

    .. info::
        :title: Shortened codes

        To create the shortened $\mathrm{RS}(n - s, k - s)$ code, construct the full-length $\mathrm{RS}(n, k)$ code,
        pass only the first $(k - s)$ message symbols to :func:`encode`, and provide only the first $(n - s)$
        received symbols to :func:`decode`. This produces the standard shortened RS construction.

    Reed-Solomon codes admit two mathematically equivalent interpretations: the BCH (cyclic, root-based)
    formulation and the polynomial evaluation formulation. This implementation follows the BCH construction,
    which realizes RS codes as cyclic codes whose generator polynomials are specified by a consecutive set of
    roots in an extension field.

    **BCH (cyclic) construction.**
    Assume $n \mid (q - 1)$, so that $\mathrm{GF}(q)$ contains a primitive $n$-th root of unity. Let
    $\alpha \in \mathrm{GF}(q)$ be such an element. A Reed-Solomon code of designed distance $d = n - k + 1$
    and starting exponent $c$ is the cyclic code whose generator polynomial has the consecutive roots

    $$
    \alpha^c,\ \alpha^{c+1},\ \ldots,\ \alpha^{c + (d - 2)}.
    $$

    Because all these roots lie in $\mathrm{GF}(q)$ itself (in contrast to BCH codes in general, which require
    $\mathrm{GF}(q^m)$), the generator polynomial is simply the product of linear factors:

    $$
    g(x) = (x - \alpha^c)\, (x - \alpha^{c+1}) \cdots (x - \alpha^{c + d - 2}).
    $$

    The resulting cyclic code has length $n$, generator degree $\deg g(x) = n - k$, and therefore dimension
    $k = n - \deg g(x)$. Its minimum distance satisfies $d = n - k + 1$, i.e., the code is MDS.

    **Evaluation-code interpretation (equivalent).**
    For completeness: RS codes may also be defined by evaluating a message polynomial
    $m(x) = m_0 + m_1 x + \cdots + m_{k-1} x^{k-1}$ at a set of $n$ distinct points in $\mathrm{GF}(q)$, typically
    $\{\alpha^0, \alpha^1, \ldots, \alpha^{n-1}\}$. In this viewpoint,

    $$
    (\, m(\alpha^0), m(\alpha^1), \ldots, m(\alpha^{n-1}) \,) \in \mathrm{GF}(q)^n
    $$

    is the codeword corresponding to $m(x)$. Choosing the evaluation points as consecutive powers of a primitive
    element yields a cyclic RS code whose generator matches the BCH construction above. Thus the BCH formulation
    used here is fully consistent with the general algebraic interpretation of Reed-Solomon codes.

    Examples:
        Construct a $\textrm{RS}(15, 9)$ code.

        .. ipython:: python

            rs = galois.ReedSolomon(15, 9); rs
            GF = rs.field; GF

        Encode a message.

        .. ipython:: python

            m = GF.Random(rs.k); m
            c = rs.encode(m); c

        Corrupt the codeword and decode the message.

        .. ipython:: python

            # Corrupt the first symbol in the codeword
            c[0] ^= 13; c
            dec_m = rs.decode(c); dec_m
            assert np.array_equal(dec_m, m)

        Instruct the decoder to return the number of corrected symbol errors.

        .. ipython:: python

            dec_m, N = rs.decode(c, errors=True); dec_m, N
            assert np.array_equal(dec_m, m)

    Group:
        fec
    """

    def __init__(
        self,
        n: int,
        k: int | None = None,
        d: int | None = None,
        field: Type[FieldArray] | None = None,
        alpha: ElementLike | None = None,
        c: int = 1,
        systematic: bool = True,
    ):
        r"""
        Constructs a general $\textrm{RS}(n, k)$ code over $\mathrm{GF}(q)$.

        Arguments:
            n: The codeword size $n$. If $n = q - 1$, the Reed-Solomon code is *primitive*.
            k: The message size $k$.

                .. important::
                    Either `k` or `d` must be provided to define the code. Both may be provided as long as they are
                    consistent.

            d: The design distance $d$. This defines the number of roots $d - 1$ in the generator
                polynomial $g(x)$ over $\mathrm{GF}(q)$. Reed-Solomon codes achieve the Singleton bound,
                so $d = n - k + 1$.
            field: The Galois field $\mathrm{GF}(q)$ that defines the alphabet of the codeword symbols.
                The default is `None` which corresponds to $\mathrm{GF}(2^m)$ where
                $2^{m - 1} \le n < 2^m$. The default field will use `matlab_primitive_poly(2, m)` for the
                irreducible polynomial.
            alpha: A primitive $n$-th root of unity $\alpha$ in $\mathrm{GF}(q)$ that defines the
                $\alpha^c, \dots, \alpha^{c+d-2}$ roots of the generator polynomial $g(x)$.
            c: The first consecutive power $c$ of $\alpha$ that defines the
                $\alpha^c, \dots, \alpha^{c+d-2}$ roots of the generator polynomial $g(x)$.
                The default is 1. If $c = 1$, the Reed-Solomon code is *narrow-sense*.
            systematic: Indicates if the encoding should be systematic, meaning the codeword is the message with
                parity appended. The default is `True`.

        See Also:
            matlab_primitive_poly, FieldArray.primitive_root_of_unity

        Examples:
            Construct a primitive, narrow-sense $\textrm{RS}(255, 223)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                galois.ReedSolomon(255, 223)
                galois.ReedSolomon(255, d=33)
                galois.ReedSolomon(255, 223, 33)

            Construct a non-primitive, narrow-sense $\textrm{RS}(85, 65)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                GF = galois.GF(2**8)
                galois.ReedSolomon(85, 65, field=GF)
                galois.ReedSolomon(85, d=21, field=GF)
                galois.ReedSolomon(85, 65, 21, field=GF)
        """
        verify_isinstance(n, int)
        verify_isinstance(k, int, optional=True)
        verify_isinstance(d, int, optional=True)
        verify_issubclass(field, FieldArray, optional=True)
        verify_isinstance(c, int)
        verify_isinstance(systematic, bool)

        if d is not None and not d >= 1:
            raise ValueError(f"Argument 'd' must be at least 1, not {d}.")
        if not c >= 0:
            raise ValueError(f"Argument 'c' must be at least 0, not {c}.")

        if field is None:
            q = 2
            m = ilog(n, q) + 1
            assert q ** (m - 1) < n + 1 <= q**m
            irreducible_poly = matlab_primitive_poly(q, m)
            field = Field(q**m, irreducible_poly=irreducible_poly)

        if alpha is None:
            alpha = field.primitive_root_of_unity(n)
        else:
            alpha = field(alpha)

        # Determine the code size from the (n, k), (n, d), or (n, k, d). Reed-Solomon codes achieve the
        # Singleton bound, so the relationship between n, k, and d is precise.
        if d is not None and k is not None:
            if not d == n - k + 1:
                raise ValueError(
                    "Arguments 'k' and 'd' were provided but are inconsistent. For Reed-Solomon codes, d = n - k + 1."
                )
        elif d is not None:
            k = n - (d - 1)
        elif k is not None:
            d = (n - k) + 1
        else:
            raise ValueError("Argument 'k' or 'd' must be provided to define the code size.")

        roots = alpha ** (c + np.arange(0, d - 1))
        generator_poly = Poly.Roots(roots)

        # Set BCH specific attributes
        self._alpha = alpha
        self._c = c
        self._is_primitive = n == field.order - 1
        self._is_narrow_sense = c == 1

        super().__init__(n, k, d, generator_poly, roots, systematic)

        # TODO: Do this?? How to standardize G and H?
        self._H = np.power.outer(roots, np.arange(n - 1, -1, -1, dtype=field.dtypes[-1]))

    def __repr__(self) -> str:
        r"""
        A terse representation of the Reed-Solomon code.

        Examples:
            Construct a primitive, narrow-sense $\textrm{RS}(255, 223)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(255, 223)
                rs

            Construct a non-primitive, narrow-sense $\textrm{RS}(85, 65)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(85, 65, field=galois.GF(2**8))
                rs
        """
        return f"<Reed-Solomon Code: [{self.n}, {self.k}, {self.d}] over {self.field.name}>"

    def __str__(self) -> str:
        r"""
        A formatted string with relevant properties of the Reed-Solomon code.

        Examples:
            Construct a primitive, narrow-sense $\textrm{RS}(255, 223)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(255, 223)
                print(rs)

            Construct a non-primitive, narrow-sense $\textrm{RS}(85, 65)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(85, 65, field=galois.GF(2**8))
                print(rs)
        """
        string = "Reed-Solomon Code:"
        string += f"\n  [n, k, d]: [{self.n}, {self.k}, {self.d}]"
        string += f"\n  field: {self.field.name}"
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

                    Encode a single message using the $\textrm{RS}(15, 9)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random(rs.k); m
                        c = rs.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = rs.encode(m, output="parity"); p

                .. md-tab-item:: Vector (shortened)

                    Encode a single message using the shortened $\textrm{RS}(11, 5)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random(rs.k - 4); m
                        c = rs.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = rs.encode(m, output="parity"); p

                .. md-tab-item:: Matrix

                    Encode a matrix of three messages using the $\textrm{RS}(15, 9)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random((3, rs.k)); m
                        c = rs.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = rs.encode(m, output="parity"); p

                .. md-tab-item:: Matrix (shortened)

                    Encode a matrix of three messages using the shortened $\textrm{RS}(11, 5)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random((3, rs.k - 4)); m
                        c = rs.encode(m); c

                    Compute the parity symbols only.

                    .. ipython:: python

                        p = rs.encode(m, output="parity"); p
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

                    Encode a single message using the $\textrm{RS}(15, 9)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random(rs.k); m
                        c = rs.encode(m); c

                    Detect no errors in the valid codeword.

                    .. ipython:: python

                        rs.detect(c)

                    Detect $d_{min}-1$ errors in the codeword.

                    .. ipython:: python

                        rs.d
                        e = GF.Random(rs.d - 1, low=1); e
                        c[0:rs.d - 1] += e; c
                        rs.detect(c)

                .. md-tab-item:: Vector (shortened)

                    Encode a single message using the shortened $\textrm{RS}(11, 5)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random(rs.k - 4); m
                        c = rs.encode(m); c

                    Detect no errors in the valid codeword.

                    .. ipython:: python

                        rs.detect(c)

                    Detect $d_{min}-1$ errors in the codeword.

                    .. ipython:: python

                        rs.d
                        e = GF.Random(rs.d - 1, low=1); e
                        c[0:rs.d - 1] += e; c
                        rs.detect(c)

                .. md-tab-item:: Matrix

                    Encode a matrix of three messages using the $\textrm{RS}(15, 9)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random((3, rs.k)); m
                        c = rs.encode(m); c

                    Detect no errors in the valid codewords.

                    .. ipython:: python

                        rs.detect(c)

                    Detect one, two, and $d_{min}-1$ errors in the codewords.

                    .. ipython:: python

                        rs.d
                        c[0, 0:1] += GF.Random(1, low=1)
                        c[1, 0:2] += GF.Random(2, low=1)
                        c[2, 0:rs.d - 1] += GF.Random(rs.d - 1, low=1)
                        c
                        rs.detect(c)

                .. md-tab-item:: Matrix (shortened)

                    Encode a matrix of three messages using the shortened $\textrm{RS}(11, 5)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random((3, rs.k - 4)); m
                        c = rs.encode(m); c

                    Detect no errors in the valid codewords.

                    .. ipython:: python

                        rs.detect(c)

                    Detect one, two, and $d_{min}-1$ errors in the codewords.

                    .. ipython:: python

                        rs.d
                        c[0, 0:1] += GF.Random(1, low=1)
                        c[1, 0:2] += GF.Random(2, low=1)
                        c[2, 0:rs.d - 1] += GF.Random(rs.d - 1, low=1)
                        c
                        rs.detect(c)
        """,
    )
    def detect(self, codeword: ArrayLike) -> bool | np.ndarray:
        return super().detect(codeword)

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        erasures: npt.NDArray | None = None,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[False] = False,
    ) -> FieldArray: ...

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        erasures: npt.NDArray | None = None,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[True] = True,
    ) -> tuple[FieldArray, int | np.ndarray]: ...

    @extend_docstring(
        _CyclicCode.decode,
        {},
        r"""
        In decoding, the syndrome vector $\mathbf{s}$ is computed by evaluating the received codeword
        $\mathbf{r}$ at the roots $\alpha^c, \dots, \alpha^{c+d-2}$ of the generator polynomial
        $g(x)$. The equivalent polynomial operation computes the remainder of $r(x)$ by $g(x)$.

        $$\mathbf{s} = [r(\alpha^c),\ \dots,\ r(\alpha^{c+d-2})] \in \mathrm{GF}(q)^{d-1}$$

        $$s(x) = r(x)\ \textrm{mod}\ g(x) \in \mathrm{GF}(q)[x]$$

        A syndrome of zeros indicates the received codeword is a valid codeword and there are no errors. If the
        syndrome is non-zero, the decoder will find an error-locator polynomial $\sigma(x)$ and the corresponding
        error locations and values.

        Examples:
            .. md-tab-set::

                .. md-tab-item:: Vector

                    Encode a single message using the $\textrm{RS}(15, 9)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random(rs.k); m
                        c = rs.encode(m); c

                    Corrupt $t$ symbols of the codeword.

                    .. ipython:: python

                        e = GF.Random(rs.t, low=1); e
                        c[0:rs.t] += e; c

                    Decode the codeword and recover the message.

                    .. ipython:: python

                        d = rs.decode(c); d
                        assert np.array_equal(d, m)

                    Decode the codeword, outputting the number of corrected errors, and recover the message.

                    .. ipython:: python

                        d, e = rs.decode(c, errors=True); d, e
                        assert np.array_equal(d, m)

                .. md-tab-item:: Vector (shortened)

                    Encode a single message using the shortened $\textrm{RS}(11, 5)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random(rs.k - 4); m
                        c = rs.encode(m); c

                    Corrupt $t$ symbols of the codeword.

                    .. ipython:: python

                        e = GF.Random(rs.t, low=1); e
                        c[0:rs.t] += e; c

                    Decode the codeword and recover the message.

                    .. ipython:: python

                        d = rs.decode(c); d
                        assert np.array_equal(d, m)

                    Decode the codeword, outputting the number of corrected errors, and recover the message.

                    .. ipython:: python

                        d, e = rs.decode(c, errors=True); d, e
                        assert np.array_equal(d, m)

                .. md-tab-item:: Matrix

                    Encode a matrix of three messages using the $\textrm{RS}(15, 9)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random((3, rs.k)); m
                        c = rs.encode(m); c

                    Corrupt the codeword. Add one error to the first codeword, two to the second, and three to the
                    third.

                    .. ipython:: python

                        c[0,0:1] += GF.Random(1, low=1)
                        c[1,0:2] += GF.Random(2, low=1)
                        c[2,0:3] += GF.Random(3, low=1)
                        c

                    Decode the codeword and recover the message.

                    .. ipython:: python

                        d = rs.decode(c); d
                        assert np.array_equal(d, m)

                    Decode the codeword, outputting the number of corrected errors, and recover the message.

                    .. ipython:: python

                        d, e = rs.decode(c, errors=True); d, e
                        assert np.array_equal(d, m)

                .. md-tab-item:: Matrix (shortened)

                    Encode a matrix of three messages using the shortened $\textrm{RS}(11, 5)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random((3, rs.k - 4)); m
                        c = rs.encode(m); c

                    Corrupt the codeword. Add one error to the first codeword, two to the second, and three to the
                    third.

                    .. ipython:: python

                        c[0,0:1] += GF.Random(1, low=1)
                        c[1,0:2] += GF.Random(2, low=1)
                        c[2,0:3] += GF.Random(3, low=1)
                        c

                    Decode the codeword and recover the message.

                    .. ipython:: python

                        d = rs.decode(c); d
                        assert np.array_equal(d, m)

                    Decode the codeword, outputting the number of corrected errors, and recover the message.

                    .. ipython:: python

                        d, e = rs.decode(c, errors=True); d, e
                        assert np.array_equal(d, m)

                .. md-tab-item:: Matrix (erasures)

                    Encode a matrix of three messages using the $\textrm{RS}(15, 9)$ code.

                    .. ipython:: python

                        rs = galois.ReedSolomon(15, 9)
                        GF = rs.field
                        m = GF.Random((3, rs.k)); m
                        c = rs.encode(m); c

                    Corrupt the codeword. Add one error to the first codeword, two to the second, and $d - 1$ to the
                    third.

                    .. ipython:: python

                        erasures = np.zeros(c.shape, dtype=bool)
                        erasures[0,0:1] = True
                        erasures[1,0:2] = True
                        erasures[2,0:rs.d - 1] = True

                        c[erasures] += GF.Random(np.sum(erasures), low=1)
                        c

                    Decode the codeword using known error locations (erasures) and recover the message.

                    .. ipython:: python

                        d = rs.decode(c, erasures=erasures); d
                        assert np.array_equal(d, m)

                    Decode the codeword, outputting the number of corrected errors, and recover the message. Notice
                    that zero unknown errors were corrected, since all errors were provided as erasures.

                    .. ipython:: python

                        d, e = rs.decode(c, erasures=erasures, errors=True); d, e
                        assert np.array_equal(d, m)
        """,
    )
    def decode(self, codeword, erasures=None, output="message", errors=False):
        return super().decode(codeword, erasures=erasures, output=output, errors=errors)

    def _decode_codeword(self, codeword: FieldArray, erasures: npt.NDArray | None) -> tuple[FieldArray, np.ndarray]:
        func = reed_solomon_decode_jit(self.field, self.field)
        dec_codeword, N_errors = func(codeword, erasures, self.n, int(self.alpha), self.c, self.roots)
        dec_codeword = dec_codeword.view(self.field)
        return dec_codeword, N_errors

    @property
    @extend_docstring(
        _CyclicCode.field,
        {},
        r"""
        Examples:
            Construct a $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.field
                print(rs.field.properties)

            Construct a $\textrm{RS}(26, 18)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(26, 18, field=galois.GF(3**3)); rs
                rs.field
                print(rs.field.properties)
        """,
    )
    def field(self) -> Type[FieldArray]:
        return super().field

    @property
    @extend_docstring(
        _CyclicCode.n,
        {},
        r"""
        Examples:
            Construct a $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.n

            Construct a $\textrm{RS}(26, 18)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(26, 18, field=galois.GF(3**3)); rs
                rs.n
        """,
    )
    def n(self) -> int:
        return super().n

    @property
    @extend_docstring(
        _CyclicCode.k,
        {},
        r"""
        Examples:
            Construct a $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.k

            Construct a $\textrm{RS}(26, 18)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(26, 18, field=galois.GF(3**3)); rs
                rs.k
        """,
    )
    def k(self) -> int:
        return super().k

    @property
    @extend_docstring(
        _CyclicCode.d,
        {},
        r"""
        Examples:
            Construct a $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.d

            Construct a $\textrm{RS}(26, 18)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(26, 18, field=galois.GF(3**3)); rs
                rs.d
        """,
    )
    def d(self) -> int:
        return super().d

    @property
    @extend_docstring(
        _CyclicCode.t,
        {},
        r"""
        Examples:
            Construct a $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.t

            Construct a $\textrm{RS}(26, 18)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(26, 18, field=galois.GF(3**3)); rs
                rs.t
        """,
    )
    def t(self) -> int:
        return super().t

    @property
    @extend_docstring(
        _CyclicCode.generator_poly,
        {},
        r"""
        Examples:
            Construct a narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$ with first
            consecutive root $\alpha$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.generator_poly
                rs.roots
                # Evaluate the generator polynomial at its roots in GF(q)
                rs.generator_poly(rs.roots)

            Construct a non-narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$ with first
            consecutive root $\alpha^3$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9, c=3); rs
                rs.generator_poly
                rs.roots
                # Evaluate the generator polynomial at its roots in GF(q)
                rs.generator_poly(rs.roots)
        """,
    )
    def generator_poly(self) -> Poly:
        return super().generator_poly

    @property
    @extend_docstring(
        _CyclicCode.parity_check_poly,
        {},
        r"""
        Examples:
            Construct a primitive $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.parity_check_poly
                rs.H

            Construct a non-primitive $\textrm{RS}(13, 9)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(13, 9, field=galois.GF(3**3)); rs
                rs.parity_check_poly
                rs.H
        """,
    )
    def parity_check_poly(self) -> Poly:
        return super().parity_check_poly

    @property
    @extend_docstring(
        _CyclicCode.roots,
        {},
        r"""
        These are consecutive powers of $\alpha^c$, specifically $\alpha^c, \dots, \alpha^{c+d-2}$.

        Examples:
            Construct a narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$ with first
            consecutive root $\alpha$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.roots
                rs.generator_poly
                # Evaluate the generator polynomial at its roots in GF(q)
                rs.generator_poly(rs.roots)

            Construct a non-narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$ with first
            consecutive root $\alpha^3$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9, c=3); rs
                rs.roots
                rs.generator_poly
                # Evaluate the generator polynomial at its roots in GF(q)
                rs.generator_poly(rs.roots)
        """,
    )
    def roots(self) -> FieldArray:
        return super().roots

    @property
    def alpha(self) -> FieldArray:
        r"""
        A primitive $n$-th root of unity $\alpha$ in $\mathrm{GF}(q)$ whose consecutive powers
        $\alpha^c, \dots, \alpha^{c+d-2}$ are roots of the generator polynomial $g(x)$.

        Examples:
            Construct a primitive $\textrm{RS}(255, 223)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(255, 223); rs
                rs.alpha
                assert rs.roots[0] == rs.alpha ** rs.c
                assert rs.alpha.multiplicative_order() == rs.n

            Construct a non-primitive $\textrm{RS}(85, 65)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(85, 65, field=galois.GF(2**8)); rs
                rs.alpha
                assert rs.roots[0] == rs.alpha ** rs.c
                assert rs.alpha.multiplicative_order() == rs.n

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
            Construct a narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$
            with first consecutive root $\alpha$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.c
                assert rs.roots[0] == rs.alpha ** rs.c
                rs.generator_poly

            Construct a narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$
            with first consecutive root $\alpha^3$. Notice the design distance is the same, however
            the generator polynomial is different.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9, c=3); rs
                rs.c
                assert rs.roots[0] == rs.alpha ** rs.c
                rs.generator_poly
        """
        return self._c

    @property
    @extend_docstring(
        _CyclicCode.G,
        {},
        r"""
        Examples:
            Construct a primitive $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.G

            Construct a non-primitive $\textrm{RS}(13, 9)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(13, 9, field=galois.GF(3**3)); rs
                rs.G

            .. ipython:: python

                rs = galois.ReedSolomon(13, 9, field=galois.GF(3**3), systematic=False); rs
                rs.G
                rs.generator_poly
        """,
    )
    def G(self) -> FieldArray:
        return super().G

    @property
    @extend_docstring(
        _CyclicCode.H,
        {},
        r"""
        Examples:
            Construct a primitive $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                rs.H
                rs.parity_check_poly

            Construct a non-primitive $\textrm{RS}(13, 9)$ code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(13, 9, field=galois.GF(3**3)); rs
                rs.H
                rs.parity_check_poly
        """,
    )
    def H(self) -> FieldArray:
        return super().H

    @property
    def is_primitive(self) -> bool:
        r"""
        Indicates if the Reed-Solomon code is *primitive*, meaning $n = q - 1$.

        Examples:
            Construct a primitive $\textrm{RS}(255, 223)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(255, 223); rs
                assert rs.is_primitive
                assert rs.n == rs.field.order - 1

            Construct a non-primitive $\textrm{RS}(85, 65)$ code over $\mathrm{GF}(2^8)$.

            .. ipython:: python

                rs = galois.ReedSolomon(85, 65, field=galois.GF(2**8)); rs
                assert not rs.is_primitive
                assert not rs.n == rs.field.order - 1
        """
        return self._is_primitive

    @property
    def is_narrow_sense(self) -> bool:
        r"""
        Indicates if the Reed-Solomon code is *narrow-sense*, meaning the roots of the generator polynomial are
        consecutive powers of $\alpha$ starting at 1, that is $\alpha, \dots, \alpha^{d-1}$.

        Examples:
            Construct a narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$
            with first consecutive root $\alpha$.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9); rs
                assert rs.is_narrow_sense
                assert rs.c == 1
                rs.generator_poly
                rs.roots

            Construct a narrow-sense $\textrm{RS}(15, 9)$ code over $\mathrm{GF}(2^4)$
            with first consecutive root $\alpha^3$. Notice the design distance is the same, however
            the generator polynomial is different.

            .. ipython:: python

                rs = galois.ReedSolomon(15, 9, c=3); rs
                assert not rs.is_narrow_sense
                assert not rs.c == 1
                rs.generator_poly
                rs.roots
        """
        return self._is_narrow_sense

    @property
    @extend_docstring(
        _CyclicCode.is_systematic,
        {},
        r"""
        Examples:
            Construct a non-primitive $\textrm{RS}(13, 9)$ systematic code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(13, 9, field=galois.GF(3**3)); rs
                assert rs.is_systematic
                rs.G

            Construct a non-primitive $\textrm{RS}(13, 9)$ non-systematic code over $\mathrm{GF}(3^3)$.

            .. ipython:: python

                rs = galois.ReedSolomon(13, 9, field=galois.GF(3**3), systematic=False); rs
                assert not rs.is_systematic
                rs.G
                rs.generator_poly
        """,
    )
    def is_systematic(self) -> bool:
        return super().is_systematic


class reed_solomon_decode_jit(bch_decode_jit):
    """
    Performs general BCH and Reed-Solomon decoding.

    References:
        - Lin, S. and Costello, D. Error Control Coding. Section 7.4.
    """

    # NOTE: Making a subclass so that these compiled functions are stored in a new namespace
