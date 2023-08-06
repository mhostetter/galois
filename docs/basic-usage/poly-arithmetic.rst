Polynomial Arithmetic
=====================

In the sections below, the finite field $\mathrm{GF}(7)$ and polynomials $f(x)$ and $g(x)$ are used.

.. ipython:: python

    GF = galois.GF(7)
    f = galois.Poly([1, 0, 4, 3], field=GF); f
    g = galois.Poly([2, 1, 3], field=GF); g

Standard arithmetic
-------------------

After creating a :doc:`polynomial over a finite field <poly>`, nearly any polynomial arithmetic operation can be
performed using Python operators. Expand any section for more details.

.. example:: Addition: `f + g`
    :collapsible:

    Add two polynomials.

    .. ipython:: python

        f
        g
        f + g

    Add a polynomial and a finite field scalar. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f + GF(3)
        GF(3) + f

.. example:: Additive inverse: `-f`
    :collapsible:

    .. ipython:: python

        f
        -f

    Any polynomial added to its additive inverse results in zero.

    .. ipython:: python

        f
        f + -f

.. example:: Subtraction: `f - g`
    :collapsible:

    Subtract one polynomial from another.

    .. ipython:: python

        f
        g
        f - g

    Subtract finite field scalar from a polynomial, or vice versa. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f - GF(3)
        GF(3) - f

.. example:: Multiplication: `f * g`
    :collapsible:

    Multiply two polynomials.

    .. ipython:: python

        f
        g
        f * g

    Multiply a polynomial and a finite field scalar. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f * GF(3)
        GF(3) * f

.. example:: Scalar multiplication: `f * 3`
    :collapsible:

    Scalar multiplication is essentially *repeated addition*. It is the "multiplication" of finite field elements
    and integers. The integer value indicates how many additions of the field element to sum.

    .. ipython:: python

        f * 4
        f + f + f + f

    In finite fields $\mathrm{GF}(p^m)$, the characteristic $p$ is the smallest value when multiplied by
    any non-zero field element that always results in 0.

    .. ipython:: python

        p = GF.characteristic; p
        f * p

.. example:: Division: `f // g`
    :collapsible:

    Divide one polynomial by another. Floor division is supported. True division is not supported since fractional polynomials are not
    currently supported.

    .. ipython:: python

        f
        g
        f // g

    Divide a polynomial by a finite field scalar, or vice versa. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f // GF(3)
        GF(3) // g

.. example:: Remainder: `f % g`
    :collapsible:

    Divide one polynomial by another and keep the remainder.

    .. ipython:: python

        f
        g
        f % g

    Divide a polynomial by a finite field scalar, or vice versa, and keep the remainder. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f % GF(3)
        GF(3) % g

.. example:: Divmod: `divmod(f, g)`
    :collapsible:

    Divide one polynomial by another and return the quotient and remainder.

    .. ipython:: python

        f
        g
        divmod(f, g)

    Divide a polynomial by a finite field scalar, or vice versa, and keep the remainder. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        divmod(f, GF(3))
        divmod(GF(3), g)

.. example:: Exponentiation: `f ** 3`
    :collapsible:

    Exponentiate a polynomial to a non-negative exponent.

    .. ipython:: python

        f
        f ** 3
        pow(f, 3)
        f * f * f

.. example:: Modular exponentiation: `pow(f, 123456789, g)`
    :collapsible:

    Exponentiate a polynomial to a non-negative exponent and reduce modulo another polynomial. This performs efficient modular exponentiation.

    .. ipython:: python

        f
        g
        # Efficiently computes (f ** 123456789) % g
        pow(f, 123456789, g)

Evaluation
----------

Polynomial objects may also be evaluated at scalars, arrays, or square matrices. Expand any section for more details.

.. example:: Evaluation (element-wise): `f(x)` or `f(X)`
    :collapsible:

    Polynomials are evaluated by invoking :func:`~galois.Poly.__call__`. They can be evaluated at scalars.

    .. ipython:: python

        f
        f(5)

        # The equivalent field calculation
        GF(5)**3 + 4*GF(5) + GF(3)

    Or they can be evaluated at arrays element-wise.

    .. ipython:: python

        x = GF([5, 6, 3, 4])

        # Evaluate f(x) element-wise at a 1-D array
        f(x)

    .. ipython:: python

        X = GF([[5, 6], [3, 4]])

        # Evaluate f(x) element-wise at a 2-D array
        f(X)

.. example:: Evaluation (square matrix): `f(X, elementwise=False)`
    :collapsible:

    Polynomials can also be evaluated at square matrices. Note, this is different than element-wise array evaluation. Here,
    the square matrix indeterminate is exponentiated using matrix multiplication. So $f(x) = x^3$ evaluated
    at the square matrix `X` equals `X @ X @ X`.

    .. ipython:: python

        f
        f(X, elementwise=False)

        # The equivalent matrix operation
        np.linalg.matrix_power(X, 3) + 4*X + GF(3)*GF.Identity(X.shape[0])

.. example:: Composition: `f(g)`
    :collapsible:

    Polynomial composition $f(g(x))$ is easily performed using an overload to :func:`~galois.Poly.__call__`.

    .. ipython:: python

        f
        g
        f(g)

Special arithmetic
------------------

Polynomial objects also work on several special arithmetic operations. Expand any section for more details.

.. example:: Greatest common denominator: `galois.gcd(f, g)`
    :collapsible:

    .. ipython:: python

        f
        g
        d = galois.gcd(f, g); d
        f % d
        g % d

    See :func:`~galois.gcd` for more details.

.. example:: Extended greatest common denominator: `galois.egcd(f, g)`
    :collapsible:

    .. ipython:: python

        f
        g
        d, s, t = galois.egcd(f, g)
        d, s, t
        f*s + g*t == d

    See :func:`~galois.egcd` for more details.

.. example:: Factor into irreducible polynomials: `galois.factors(f) == f.factors()`
    :collapsible:

    .. ipython:: python

        f
        galois.factors(f)
        f.factors()

    See :func:`~galois.factors` or :func:`galois.Poly.factors` for more details.
