Getting Started
===============

The :doc:`getting-started` guide is intended to assist the user with installing the library, creating two example
arrays, and performing basic array arithmetic. See :doc:`Basic Usage </basic-usage/array-classes>` for more detailed discussions
and examples.

Install the package
-------------------

The latest version of :obj:`galois` can be installed from `PyPI <https://pypi.org/project/galois/>`_ using `pip`.

.. code-block:: console

   $ python3 -m pip install galois

Import the :obj:`galois` package in Python.

.. ipython:: python

   import galois
   galois.__version__

Create a :obj:`~galois.FieldArray` subclass
-------------------------------------------

Next, create a :obj:`~galois.FieldArray` subclass for the specific finite field you'd like to work in. This is created using
the :func:`~galois.GF` class factory. In this example, we are working in $\mathrm{GF}(3^5)$.

.. ipython:: python

   GF = galois.GF(3**5)
   print(GF.properties)

The :obj:`~galois.FieldArray` subclass `GF` is a subclass of :obj:`~numpy.ndarray` that performs all arithmetic in the Galois field
$\mathrm{GF}(3^5)$, not in $\mathbb{R}$.

.. ipython:: python

   issubclass(GF, galois.FieldArray)
   issubclass(GF, np.ndarray)

See :doc:`/basic-usage/array-classes` for more details.

Create two :obj:`~galois.FieldArray` instances
----------------------------------------------

Next, create a new :obj:`~galois.FieldArray` `x` by passing an :obj:`~galois.typing.ArrayLike` object to `GF`'s constructor.

.. ipython:: python

   x = GF([236, 87, 38, 112]); x

The array `x` is an instance of :obj:`~galois.FieldArray` and also an instance of :obj:`~numpy.ndarray`.

.. ipython:: python

   isinstance(x, galois.FieldArray)
   isinstance(x, np.ndarray)

Create a second :obj:`~galois.FieldArray` `y` by converting an existing NumPy array (without copying it) by invoking
`.view()`. When finished working in the finite field, view it back as a NumPy array with `.view(np.ndarray)`.

.. ipython:: python

   # y represents an array created elsewhere in the code
   y = np.array([109, 17, 108, 224]); y
   y = y.view(GF); y

See :doc:`/basic-usage/array-creation` for more details.

Change the element representation
---------------------------------

The representation of finite field elements can be set to either the integer (`"int"`), polynomial (`"poly"`),
or power (`"power"`) representation. The default representation is the integer representation since integers are natural when
working with integer NumPy arrays.

Set the element representation by passing the `repr` keyword argument to :func:`~galois.GF` or by calling the :func:`~galois.FieldArray.repr`
classmethod. Choose whichever element representation is most convenient.

.. ipython:: python

   # The default is the integer representation
   x
   GF.repr("poly"); x
   GF.repr("power"); x
   # Reset to the integer representation
   GF.repr("int");

See :doc:`/basic-usage/element-representation` for more details.

Perform array arithmetic
------------------------

Once you have two Galois field arrays, nearly any arithmetic operation can be performed using normal NumPy arithmetic.
The traditional `NumPy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ apply.

Standard element-wise array arithmetic -- addition, subtraction, multiplication, and division -- are easily preformed.

.. ipython:: python

   x + y
   x - y
   x * y
   x / y

More complicated arithmetic, like square root and logarithm base $\alpha$, are also supported.

.. ipython:: python

   np.sqrt(x)
   np.log(x)

See :doc:`/basic-usage/array-arithmetic` for more details.
