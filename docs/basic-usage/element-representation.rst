Element Representation
======================

The representation of finite field elements can be set to either their integer (`"int"`), polynomial (`"poly"`),
or power (`"power"`) representation.

In prime fields $\mathrm{GF}(p)$, elements are integers in $\{0, 1, \dots, p-1\}$. Their two useful representations
are the integer and power representation.

In extension fields $\mathrm{GF}(p^m)$, elements are polynomials over $\mathrm{GF}(p)$ with degree less than $m$.
All element representations are useful. The polynomial representation allows *proper* representation of the element as a polynomial
over its prime subfield. However, the integer representation is more compact for displaying large arrays.

Set the element representation
------------------------------

The field element representation can be set during :obj:`~galois.FieldArray` subclass creation by passing the `repr` keyword
argument to the :func:`~galois.GF` class factory.

.. ipython:: python

    GF = galois.GF(3**5, repr="poly")
    x = GF([17, 4])
    x
    print(x)

The current element representation is accessed with the :obj:`~galois.FieldArray.element_repr` class property.

.. ipython:: python

    GF.element_repr

The element representation can be temporarily changed using the :func:`~galois.FieldArray.repr` classmethod as a context manager.

.. ipython:: python

    # Inside the context manager, x prints using the power representation
    with GF.repr("power"):
        print(x)

    # Outside the context manager, x prints using the previous representation
    print(x)

The element representation can be permanently changed using the :func:`~galois.FieldArray.repr` classmethod (not as a context manager).

.. ipython:: python

    # The old polynomial representation
    x

    GF.repr("int");

    # The new integer representation
    x

.. _int-repr:

Integer representation
----------------------

The integer representation (the default) displays all finite field elements as integers in $\{0, 1, \dots, p^m-1\}$.

In prime fields, the integer representation is simply the integer element in $\{0, 1, \dots, p-1\}$.

.. ipython:: python

    GF = galois.GF(31)
    GF(11)

In extension fields, the integer representation converts an element's degree-$m-1$ polynomial over $\mathrm{GF}(p)$ into
its integer equivalent. The integer equivalent of a polynomial is a radix-$p$ integer of its coefficients, with the highest-degree
coefficient as the most-significant digit and zero-degree coefficient as the least-significant digit.

.. ipython:: python

    GF = galois.GF(3**5)
    GF(17)
    GF("x^2 + 2x + 2")
    # Integer/polynomial equivalence
    p = 3; p**2 + 2*p + 2 == 17

.. _poly-repr:

Polynomial representation
-------------------------

The polynomial representation displays all finite field elements as polynomials over their prime subfield with degree less than $m$.

In prime fields $m = 1$, therefore the polynomial representation is equivalent to the integer representation because the
polynomials all have degree 0.

.. ipython:: python

    GF = galois.GF(31, repr="poly")
    GF(11)

In extension fields, the polynomial representation displays the elements naturally as polynomials over their prime subfield.
This is useful, however it can become cluttered for large arrays.

.. ipython:: python

    GF = galois.GF(3**5, repr="poly")
    GF(17)
    GF("x^2 + 2x + 2")
    # Integer/polynomial equivalence
    p = 3; p**2 + 2*p + 2 == 17

.. tip::

    Use :func:`~galois.set_printoptions` to display the polynomial coefficients in degree-ascending order.
    Use :func:`numpy.set_printoptions` to increase the line width to display large arrays more clearly. See :ref:`numpy-print-options`
    for more details.

.. _power-repr:

Power representation
--------------------

The power representation displays all finite field elements as powers of the field's primitive element $\alpha$.

.. slow-performance::

    To display elements in the power representation, :obj:`galois` must compute the discrete logarithm of each element
    displayed. For large fields or fields using :ref:`explicit calculation <explicit-calculation>`, this process can
    take a while. However, when using :ref:`lookup tables <lookup-tables>` this representation is just as fast as
    the others.

In prime fields, the elements are displayed as $\{0, 1, \alpha, \alpha^2, \dots, \alpha^{p-2}\}$.

.. ipython:: python

    GF = galois.GF(31, repr="power")
    GF(11)

.. ipython:: python

    GF.repr("int");
    alpha = GF.primitive_element; alpha
    alpha ** 23

In extension fields, the elements are displayed as $\{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m-2}\}$.

.. ipython:: python

    GF = galois.GF(3**5, repr="power")
    GF(17)

.. ipython:: python

    GF.repr("int");
    alpha = GF.primitive_element; alpha
    alpha ** 222

Vector representation
---------------------

The vector representation, while not a valid input to :func:`~galois.FieldArray.repr`, represents finite field elements
as vectors of their polynomial coefficients.

The vector representation is accessed using the :func:`~galois.FieldArray.vector` method.

.. ipython:: python

    GF = galois.GF(3**5, repr="poly")
    GF("x^2 + 2x + 2")
    GF("x^2 + 2x + 2").vector()

An N-D array over $\mathrm{GF}(p^m)$ is converted to a (N + 1)-D array over $\mathrm{GF}(p)$ with the added dimension having
size $m$. The first value of the vector is the highest-degree coefficient.

.. ipython:: python

    GF(["x^2 + 2x + 2", "2x^4 + x"])
    GF(["x^2 + 2x + 2", "2x^4 + x"]).vector()

Arrays can be created from the vector representation using the :func:`~galois.FieldArray.Vector` classmethod.

.. ipython:: python

    GF.Vector([[0, 0, 1, 2, 2], [2, 0, 0, 1, 0]])

.. _numpy-print-options:

NumPy print options
-------------------

NumPy displays arrays with a default line width of 75 characters. This is problematic for large arrays. It is especially problematic
for arrays using the polynomial representation, where each element occupies a lot of space. This can be changed by modifying
NumPy's print options.

For example, below is a $5 \times 5$ matrix over $\mathrm{GF}(3^5)$ displayed in the polynomial representation.
With the default line width, the array is quite difficult to read.

.. ipython:: python

    GF = galois.GF(3**5, repr="poly")
    x = GF.Random((5, 5)); x

The readability is improved by increasing the line width using :func:`numpy.set_printoptions`.

.. ipython:: python

    @suppress
    width = np.get_printoptions()["linewidth"]
    np.set_printoptions(linewidth=200)
    x
    @suppress
    np.set_printoptions(linewidth=width)
    @suppress
    GF.repr("int");

Representation comparisons
--------------------------

For any finite field, each of the four element representations can be easily compared using the :func:`~galois.FieldArray.repr_table` classmethod.

.. ipython:: python

    GF = galois.GF(3**3)
    print(GF.repr_table())
