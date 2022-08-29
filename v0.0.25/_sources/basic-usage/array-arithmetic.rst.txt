Array Arithmetic
================

After creating a :ref:`Galois field array class` and one or two :ref:`Galois field arrays <Galois field array>`,
nearly any arithmetic operation can be performed using normal NumPy ufuncs or Python operators.

In the sections below, the finite field :math:`\mathrm{GF}(3^5)` and arrays :math:`x` and :math:`y` are used.

.. ipython:: python

    GF = galois.GF(3**5)
    x = GF([184, 25, 157, 31]); x
    y = GF([179, 9, 139, 27]); y

Ufuncs
------

`NumPy ufuncs <https://numpy.org/devdocs/reference/ufuncs.html>`_ are universal functions that operate on scalars. Unary ufuncs operate on
a single scalar and binary ufuncs operate on two scalars. NumPy extends the scalar operation of ufuncs to operate on arrays in various ways.
This extensibility enables `NumPy broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

Expand any section for more details.

.. details:: Addition: `x + y == np.add(x, y)`

    .. ipython:: python

        x + y
        np.add(x, y)

.. details:: Additive inverse: `-x == np.negative(x)`

    .. ipython:: python

        -x
        np.negative(x)

    Any array added to its additive inverse results in zero.

    .. ipython:: python

        x + np.negative(x)

.. details:: Subtraction: `x - y == np.subtract(x, y)`

    .. ipython:: python

        x - y
        np.subtract(x, y)

.. details:: Multiplication: `x * y == np.multiply(x, y)`

    .. ipython:: python

        x * y
        np.multiply(x, y)

.. details:: Scalar multiplication: `x * z == np.multiply(x, z)`

    Scalar multiplication is essentially *repeated addition*. It is the "multiplication" of finite field elements
    and integers. The integer value indicates how many additions of the field element to sum.

    .. ipython:: python

        x * 4
        np.multiply(x, 4)
        x + x + x + x

    In finite fields :math:`\mathrm{GF}(p^m)`, the characteristic :math:`p` is the smallest value when multiplied by
    any non-zero field element that results in :math:`0`.

    .. ipython:: python

        p = GF.characteristic; p
        x * p

.. details:: Multiplicative inverse: `y ** -1 == np.reciprocal(y)`

    .. ipython:: python

        y ** -1
        GF(1) / y
        np.reciprocal(y)

    Any array multiplied by its multiplicative inverse results in one.

    .. ipython:: python

        y * np.reciprocal(y)

.. details:: Division: `x / y == x // y == np.divide(x, y)`

    .. ipython:: python

        x / y
        x // y
        np.divide(x, y)

.. details:: Remainder: `x % y == np.remainder(x, y)`

    .. ipython:: python

        x % y
        np.remainder(x, y)

.. details:: Divmod: `divmod(x, y) == np.divmod(x, y)`

    .. ipython:: python

        x / y, x % y
        divmod(x, y)
        np.divmod(x, y)

    .. ipython:: python

        q, r = divmod(x, y)
        q*y + r == x

.. details:: Exponentiation: `x ** z == np.power(x, z)`

    .. ipython:: python

        x ** 3
        np.power(x, 3)
        x * x * x

.. details:: Square root: `np.sqrt(x)`

    .. ipython:: python

        # Ensure the elements of x have square roots
        x.is_quadratic_residue()
        z = np.sqrt(x); z
        z ** 2 == x

.. details:: Logarithm: `np.log(x)`

    .. ipython:: python

        z = np.log(y); z
        α = GF.primitive_element; α
        α ** z == y

Ufunc methods
-------------

*Galois field arrays* support `NumPy ufunc methods <https://numpy.org/devdocs/reference/ufuncs.html#methods>`_. Ufunc methods allow
a user to apply a NumPy ufunc in a unique way across the target array. All arithmetic ufuncs are supported.

Expand any section for more details.

.. details:: `reduce()`

    The :obj:`numpy.ufunc.reduce` methods reduce the input array's dimension by one, applying the ufunc across one axis.

    .. ipython:: python

        np.add.reduce(x)
        x[0] + x[1] + x[2] + x[3]

    .. ipython:: python

        np.multiply.reduce(x)
        x[0] * x[1] * x[2] * x[3]

.. details:: `accumulate()`

    The :obj:`numpy.ufunc.accumulate` methods accumulate the result of the ufunc across a specified axis.

    .. ipython:: python

        np.add.accumulate(x)
        GF([x[0], x[0] + x[1], x[0] + x[1] + x[2], x[0] + x[1] + x[2] + x[3]])

    .. ipython:: python

        np.multiply.accumulate(x)
        GF([x[0], x[0] * x[1], x[0] * x[1] * x[2], x[0] * x[1] * x[2] * x[3]])

.. details:: `reduceat()`

    The :obj:`numpy.ufunc.reduceat` methods reduces the input array's dimension by one, applying the ufunc across one axis
    in-between certain indices.

    .. ipython:: python

        np.add.reduceat(x, [0, 3])
        GF([x[0] + x[1] + x[2], x[3]])

    .. ipython:: python

        np.multiply.reduceat(x, [0, 3])
        GF([x[0] * x[1] * x[2], x[3]])

.. details:: `outer()`

    The :obj:`numpy.ufunc.outer` methods applies the ufunc to each pair of inputs.

    .. ipython:: python

        np.add.outer(x, y)

    .. ipython:: python

        np.multiply.outer(x, y)

.. details:: `at()`

    The :obj:`numpy.ufunc.at` methods performs the ufunc in-place at the specified indices.

    .. ipython:: python

        z = x.copy()
        # Negate indices 0 and 1 in-place
        np.negative.at(x, [0, 1]); x
        z[0:1] *= -1; z

Advanced arithmetic
-------------------

.. details:: Convolution: `np.convolve(x, y)`

    .. ipython:: python

        np.convolve(x, y)
