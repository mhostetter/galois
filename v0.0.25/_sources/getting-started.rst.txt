Getting Started
===============

The :ref:`Getting Started` guide is intended to assist the user in installing the library, creating two example
arrays, and performing basic array arithmetic. See :ref:`Basic Usage <Galois Field Classes>` for more detailed discussions
and examples.

Install the package
-------------------

The latest version of :obj:`galois` can be installed from `PyPI <https://pypi.org/project/galois/>`_ using `pip`.

.. code-block:: sh

   $ python3 -m pip install galois

Import the :obj:`galois` package in Python.

.. ipython:: python

   import galois
   galois.__version__

Create a Galois field array class
---------------------------------

Next, create a :ref:`Galois field array class` for the specific finite field you'd like to work in. This is created using
the :func:`galois.GF` class factory. In this example, we are working in :math:`\mathrm{GF}(2^8)`.

.. ipython:: python

   GF = galois.GF(2**8)
   GF
   print(GF)

The *Galois field array class* `GF` is a subclass of :obj:`numpy.ndarray` that performs all arithmetic in the Galois field
:math:`\mathrm{GF}(2^8)`, not in :math:`\mathbb{R}`.

.. ipython:: python

   issubclass(GF, np.ndarray)

See :ref:`Galois Field Classes` for more details.

Create two Galois field arrays
------------------------------

Next, create a new :ref:`Galois field array` `x` by passing an :ref:`array-like object <Create a new array>` to the
*Galois field array class* `GF`.

.. ipython:: python

   x = GF([45, 36, 7, 74, 135]); x

Create a second *Galois field array* `y` by converting an existing NumPy array (without copying it) by invoking `.view()`. When finished
working in the finite field, view it back as a NumPy array with `.view(np.ndarray)`.

.. ipython:: python

   # y represents an array created elsewhere in the code
   y = np.array([103, 146, 186, 83, 112], dtype=int); y
   y = y.view(GF); y

The *Galois field array* `x` is an instance of the *Galois field array class* `GF` (and also an instance of :obj:`numpy.ndarray`).

.. ipython:: python

   isinstance(x, GF)
   isinstance(x, np.ndarray)

See :ref:`Array Creation` for more details.

Change the element representation
---------------------------------

The display representation of finite field elements can be set to either the integer (`"int"`), polynomial (`"poly"`),
or power (`"power"`) representation. The default representation is the integer representation since that is natural when
working with integer NumPy arrays.

Set the display mode by passing the `display` keyword argument to :func:`galois.GF` or by calling the :func:`galois.FieldClass.display` method.
Choose whichever element representation is most convenient for you.

.. ipython:: python

   # The default representation is the integer representation
   x
   GF.display("poly"); x
   GF.display("power"); x
   # Reset to the integer representation
   GF.display("int");

See :ref:`Field Element Representation` for more details.

Perform array arithmetic
------------------------

Once you have two Galois field arrays, nearly any arithmetic operation can be performed using normal NumPy arithmetic.
The traditional `NumPy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ apply.

Standard element-wise array arithmetic -- like addition, subtraction, multiplication, and division -- are easily preformed.

.. ipython:: python

   x + y
   x - y
   x * y
   x / y

More complicated arithmetic, like square root and logarithm base :math:`\alpha`, are also supported.

.. ipython:: python

   np.sqrt(x)
   np.log(x)

See :ref:`Array Arithmetic` for more details.
