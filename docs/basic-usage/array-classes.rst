Array Classes
=============

The :obj:`galois` library subclasses :obj:`~numpy.ndarray` to provide arithmetic over Galois fields and rings (future).

:obj:`~galois.Array` subclasses
-------------------------------

The main abstract base class is :obj:`~galois.Array`. It has two abstract subclasses: :obj:`~galois.FieldArray` and
:obj:`~galois.RingArray` (future). None of these abstract classes may be instantiated directly. Instead, specific
subclasses for :math:`\mathrm{GF}(p^m)` and :math:`\mathrm{GR}(p^e, m)` are created at runtime with :func:`~galois.GF`
and :func:`~galois.GR` (future).

:obj:`~galois.FieldArray` subclasses
------------------------------------

A :obj:`~galois.FieldArray` subclass is created using the class factory function :func:`~galois.GF`.

.. ipython:: python

    GF = galois.GF(3**5)
    print(GF.properties)

The `GF` class is a subclass of :obj:`~galois.FieldArray` and a subclasses of :obj:`~numpy.ndarray`.

.. ipython:: python

    issubclass(GF, galois.FieldArray)
    issubclass(GF, galois.Array)
    issubclass(GF, np.ndarray)

Class singletons
................

:obj:`~galois.FieldArray` subclasses of the same type (order, irreducible polynomial, and primitive element) are singletons.

Here is the creation (twice) of the field :math:`\mathrm{GF}(3^5)` defined with the default irreducible
polynomial :math:`x^5 + 2x + 1`. They *are* the same class (a singleton), not just equivalent classes.

.. ipython:: python

    galois.GF(3**5) is galois.GF(3**5)

The expense of class creation is incurred only once. So, subsequent calls of `galois.GF(3**5)` are extremely inexpensive.

However, the field :math:`\mathrm{GF}(3^5)` defined with irreducible polynomial :math:`x^5 + x^2 + x + 2`, while isomorphic to the
first field, has different arithmetic. As such, :func:`~galois.GF` returns a unique :obj:`~galois.FieldArray` subclass.

.. ipython:: python

    galois.GF(3**5) is galois.GF(3**5, irreducible_poly="x^5 + x^2 + x + 2")

Methods and properties
......................

All of the methods and properties related to :math:`\mathrm{GF}(p^m)`, not one of its arrays, are documented as class methods
and class properties in :obj:`~galois.FieldArray`. For example, the irreducible polynomial of the finite field is accessed
with :obj:`~galois.FieldArray.irreducible_poly`.

.. ipython:: python

    GF.irreducible_poly

:obj:`~galois.FieldArray` instances
-----------------------------------

A :obj:`~galois.FieldArray` instance is created using `GF`'s constructor.

.. ipython:: python

    x = GF([23, 78, 163, 124])
    x

The array `x` is an instance of :obj:`~galois.FieldArray` and also an instance of :obj:`~numpy.ndarray`.

.. ipython:: python

    isinstance(x, GF)
    isinstance(x, galois.FieldArray)
    isinstance(x, galois.Array)
    isinstance(x, np.ndarray)

The :obj:`~galois.FieldArray` subclass is easily recovered from a :obj:`~galois.FieldArray` instance using :func:`type`.

.. ipython:: python

    type(x) is GF

Constructors
............

Several classmethods are defined in :obj:`~galois.FieldArray` that function as alternate constructors. By convention,
alternate constructors use `PascalCase` while other classmethods use `snake_case`.

For example, to generate a random array of given shape call :func:`~galois.FieldArray.Random`.

.. ipython:: python

    GF.Random((2, 3))

Or, create an identity matrix using :func:`~galois.FieldArray.Identity`.

.. ipython:: python

    GF.Identity(4)

Methods
.......

All of the methods that act on :obj:`~galois.FieldArray` instances are documented as instance methods in :obj:`~galois.FieldArray`.
For example, the multiplicative order of each finite field element is calculated using :func:`~galois.FieldArray.multiplicative_order`.

.. ipython:: python

    x.multiplicative_order()
