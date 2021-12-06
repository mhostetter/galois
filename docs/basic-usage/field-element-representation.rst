Field Element Representation
============================

The display representation of finite field elements can be set to either their integer (`"int"`), polynomial (`"poly"`),
or power (`"power"`) representation.

In prime fields :math:`\mathrm{GF}(p)`, elements are integers in :math:`\{0, 1, \dots, p-1\}`. Their two useful representations
are the integer and power representation.

In extension fields :math:`\mathrm{GF}(p^m)`, elements are polynomials over :math:`\mathrm{GF}(p)` with degree less than :math:`m`.
All display representations are useful. The polynomial representation allows *proper* representation of the element as a polynomial
over its prime subfield. However, the integer representation is more compact for displaying large arrays.

Setting the display mode
------------------------

The field element display mode can be set at :ref:`Galois field array class` construction by passing the `display` keyword
argument to the :func:`galois.GF` class factory.

.. ipython:: python

    GF = galois.GF(3**5, display="poly")
    x = GF([17, 4]); x

The current display mode is accessible with the :obj:`galois.FieldClass.display_mode` property.

.. ipython:: python

    GF.display_mode

The display mode can be temporarily changed using the :func:`galois.FieldClass.display` method as a context manager.

.. ipython:: python

    # Inside the context manager, x prints using the power representation
    with GF.display("power"):
        print(x)

    # Outside the context manager, x prints using the previous representation
    print(x)

The display mode can be permanently changed using the :func:`galois.FieldClass.display` method.

.. ipython:: python

    # The old polynomial display mode
    x

    GF.display("int");

    # The new integer display mode
    x

Integer representation
----------------------

The integer display mode (the default) displays all finite field elements as integers in :math:`\{0, 1, \dots, p^m-1\}`.

In prime fields, the integer representation is simply the integer element in :math:`\{0, 1, \dots, p-1\}`.

.. ipython:: python

    GF = galois.GF(31)
    GF(11)

In extension fields, the integer representation converts and element's degree-:math:`m-1` polynomial over :math:`\mathrm{GF}(p)` into
its integer equivalent. The integer equivalent of a polynomial is a radix-:math:`p` integer of its coefficients, with the highest-degree
coefficient as the most-significant digit and zero-degree coefficient as the least-significant digit.

.. ipython:: python

    GF = galois.GF(3**5)
    GF(17)
    GF("α^2 + 2α + 2")
    # Integer/polynomial equivalence
    p = 3; p**2 + 2*p + 2 == 17

Polynomial representation
-------------------------

The polynomial display mode displays all finite field elements as polynomials over their prime subfield with degree less than :math:`m`.

In prime fields, :math:`m = 1` and, therefore, the polynomial representation is equivalent to the integer representation because the
polynomials all have degree :math:`0`.

.. ipython:: python

    GF = galois.GF(31, display="poly")
    GF(11)

In extension fields, the polynomial representation displays the elements naturally as polynomials over their prime subfield.
This is useful, however it can become cluttered for large arrays.

.. ipython:: python

    GF = galois.GF(3**5, display="poly")
    GF(17)
    GF("α^2 + 2α + 2")
    # Integer/polynomial equivalence
    p = 3; p**2 + 2*p + 2 == 17

Power representation
--------------------

The power display mode represents the elements as powers of the finite field's primitive element :math:`\alpha`.

In prime fields, the elements are displayed as :math:`\{0, \alpha, \alpha^2, \dots, \alpha^{p-2}\}`.

.. ipython:: python

    GF = galois.GF(31, display="power")
    GF(11)

.. ipython:: python

    GF.display("int");
    α = GF.primitive_element; α
    α**23

In extension fields, the elements are displayed as :math:`\{0, \alpha, \alpha^2, \dots, \alpha^{p^m-2}\}`.

.. ipython:: python

    GF = galois.GF(3**5, display="power")
    GF(17)

.. ipython:: python

    GF.display("int");
    α = GF.primitive_element; α
    α**222
