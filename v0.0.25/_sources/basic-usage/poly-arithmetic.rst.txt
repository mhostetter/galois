Polynomial Arithmetic
=====================

Standard arithmetic
-------------------

After creating a :doc:`polynomial over a finite field <poly>`, nearly any polynomial arithmetic operation can be
performed using Python operators.

In the sections below, the finite field :math:`\mathrm{GF}(7)` and polynomials :math:`f(x)` and :math:`g(x)` are used.

.. ipython:: python

    GF = galois.GF(7)
    f = galois.Poly([1, 0, 4, 3], field=GF); f
    g = galois.Poly([2, 1, 3], field=GF); g

Expand any section for more details.

.. details:: Addition: `f + g`

    Add two polynomials.

    .. ipython:: python

        f + g

    Add a polynomial and a finite field scalar. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f + GF(3)
        GF(3) + f

.. details:: Additive inverse: `-f`

    .. ipython:: python

        -f

    Any polynomial added to its additive inverse results in zero.

    .. ipython:: python

        f + -f

.. details:: Subtraction: `f - g`

    Subtract one polynomial from another.

    .. ipython:: python

        f - g

    Subtract finite field scalar from a polynomial, or vice versa. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f - GF(3)
        GF(3) - f

.. details:: Multiplication: `f * g`

    Multiply two polynomials.

    .. ipython:: python

        f * g

    Multiply a polynomial and a finite field scalar. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f * GF(3)
        GF(3) * f

.. details:: Scalar multiplication: `f * 3`

    Scalar multiplication is essentially *repeated addition*. It is the "multiplication" of finite field elements
    and integers. The integer value indicates how many additions of the field element to sum.

    .. ipython:: python

        f * 4
        f + f + f + f

    In finite fields :math:`\mathrm{GF}(p^m)`, the characteristic :math:`p` is the smallest value when multiplied by
    any non-zero field element that always results in 0.

    .. ipython:: python

        p = GF.characteristic; p
        f * p

.. details:: Division: `f // g`

    Divide one polynomial by another. Floor division is supported. True division is not supported since fractional polynomials are not
    currently supported.

    .. ipython:: python

        f // g

    Divide a polynomial by a finite field scalar, or vice versa. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f // GF(3)
        GF(3) // g

.. details:: Remainder: `f % g`

    Divide one polynomial by another and keep the remainder.

    .. ipython:: python

        f % g

    Divide a polynomial by a finite field scalar, or vice versa, and keep the remainder. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        f % GF(3)
        GF(3) % g

.. details:: Divmod: `divmod(f, g)`

    Divide one polynomial by another and return the quotient and remainder.

    .. ipython:: python

        divmod(f, g)

    Divide a polynomial by a finite field scalar, or vice versa, and keep the remainder. The scalar is treated as a 0-degree polynomial.

    .. ipython:: python

        divmod(f, GF(3))
        divmod(GF(3), g)

.. details:: Exponentiation: `f ** 3`

    Exponentiate a polynomial to a non-negative exponent.

    .. ipython:: python

        f ** 3
        pow(f, 3)
        f * f * f

.. details:: Modular exponentiation: `pow(f, 123456789, g)`

    Exponentiate a polynomial to a non-negative exponent and reduce modulo another polynomial. This performs efficient modular exponentiation.

    .. ipython:: python

        # Efficiently computes (f ** 123456789) % g
        pow(f, 123456789, g)

Special arithmetic
------------------

Polynomial objects also work on several special arithmetic operations. Below are some examples.

.. ipython:: python

    GF = galois.GF(31)
    f = galois.Poly([1, 30, 0, 26, 6], field=GF); f
    g = galois.Poly([4, 17, 3], field=GF); g

Compute the polynomial greatest common divisor using :func:`~galois.gcd` and :func:`~galois.egcd`.

.. ipython:: python

    galois.gcd(f, g)
    galois.egcd(f, g)

Factor a polynomial into its irreducible polynomial factors using :func:`~galois.factors`.

.. ipython:: python

    galois.factors(f)

Polynomial evaluation
---------------------

Polynomials are evaluated by invoking :func:`~galois.Poly.__call__`. They can be evaluated at scalars.

.. ipython:: python

    GF = galois.GF(31)
    f = galois.Poly([1, 0, 0, 15], field=GF); f
    f(26)

    # The equivalent field calculation
    GF(26)**3 + GF(15)

Or they can be evaluated at arrays element-wise.

.. ipython:: python

    x = GF([26, 13, 24, 4])

    # Evaluate f(x) element-wise at a 1-D array
    f(x)

.. ipython:: python

    X = GF([[26, 13], [24, 4]])

    # Evaluate f(x) element-wise at a 2-D array
    f(X)

Or they can also be evaluated at square matrices. Note, this is different than element-wise array evaluation. Here,
the square matrix indeterminate is exponentiated using matrix multiplication. So :math:`f(x) = x^3` evaluated
at the square matrix `X` equals `X @ X @ X`.

.. ipython:: python

    f

    # Evaluate f(x) at the 2-D square matrix
    f(X, elementwise=False)

    # The equivalent matrix operation
    np.linalg.matrix_power(X, 3) + GF(15)*GF.Identity(X.shape[0])
