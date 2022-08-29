Galois Field Classes
====================

There are two key classes in the :obj:`galois` library. Understanding them and their relationship will
make using the library and accessing the appropriate documentation easier. These two classes are the
:obj:`galois.FieldClass` metaclass and :obj:`galois.FieldArray`.

The documentation also regularly refers to a :ref:`Galois field array class` and a :ref:`Galois field array`.
Both terms are defined on this page.

Galois field array class
------------------------

A *Galois field array class* is created using the class factory function :func:`galois.GF`.

.. ipython:: python

    GF = galois.GF(3**5)
    GF
    print(GF)

The *Galois field array class* `GF` is a subclass of :obj:`galois.FieldArray` (which itself subclasses :obj:`numpy.ndarray`) and
has :obj:`galois.FieldClass` as its metaclass.

.. ipython:: python

    issubclass(GF, np.ndarray)
    issubclass(GF, galois.FieldArray)
    isinstance(GF, galois.FieldClass)

Methods and properties
......................

All of the methods and properties related to the Galois field itself, not one of its arrays, are documented in :obj:`galois.FieldClass`.
For example, the irreducible polynomial of the finite field is accessed with :obj:`galois.FieldClass.irreducible_poly`.

.. ipython:: python

    GF.irreducible_poly

Class singletons
................

*Galois field array classes* of the same order with the same irreducible polynomial are singletons.

Here is the creation (twice) of the field :math:`\mathrm{GF}(3^5)` defined with the default irreducible
polynomial :math:`x^5 + 2x + 1`. They *are* the same class (a singleton), not just equivalent classes.

.. ipython:: python

    galois.GF(3**5) is galois.GF(3**5)

The expense of class creation is incurred only once. So, subsequent calls of `galois.GF(3**5)` are extremely inexpensive.

However, the field :math:`\mathrm{GF}(3^5)` defined with irreducible polynomial :math:`x^5 + x^2 + x + 2`, while isomorphic to the
first field, has different arithmetic. As such, :func:`galois.GF` returns a unique *Galois field array class*.

.. ipython:: python

    galois.GF(3**5) is galois.GF(3**5, irreducible_poly="x^5 + x^2 + x + 2")

Galois field array
------------------

A *Galois field array* is created using the constructor of the *Galois field array class* `GF`.

.. ipython:: python

    x = GF([23, 78, 163, 124])
    x

The *Galois field array* `x` is an instance of the *Galois field array class* `GF`. Since `GF` subclasses :obj:`numpy.ndarray`,
`x` is also an instance of :obj:`numpy.ndarray`.

.. ipython:: python

    isinstance(x, np.ndarray)
    isinstance(x, GF)

A *Galois field array class* is easily recovered from a *Galois field array* using :func:`type`.

.. ipython:: python

    type(x) is GF

Methods
.......

All of the methods that act on *Galois field arrays* are documented in :obj:`galois.FieldArray`. For example, the multiplicative order
of each finite field element is calculated using :func:`galois.FieldArray.multiplicative_order`.

.. ipython:: python

    x.multiplicative_order()

Or, convert an N-D array over :math:`\mathrm{GF}(3^5)` to an (N + 1)-D array of its polynomial coefficients over :math:`\mathrm{GF}(3)`
using :func:`galois.FieldArray.vector`.

.. ipython:: python

    x.vector()

Classmethods
............

Several `classmethods <https://docs.python.org/3/library/functions.html#classmethod>`_ are defined in :obj:`galois.FieldArray`. These methods
produce *Galois field arrays*. By convention, classmethods use `PascalCase`, while methods use `snake_case`.

For example, to generate a random array of given shape call :func:`galois.FieldArray.Random`.

.. ipython:: python

    GF.Random((2, 3))
