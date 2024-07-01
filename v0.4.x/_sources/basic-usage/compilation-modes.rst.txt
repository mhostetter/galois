Compilation Modes
=================

The :obj:`galois` library supports finite field arithmetic on NumPy arrays by just-in-time compiling custom
`NumPy ufuncs <https://numpy.org/doc/stable/reference/ufuncs.html>`_. It uses `Numba <https://numba.pydata.org/>`_ to JIT
compile ufuncs written in pure Python. The created :obj:`~galois.FieldArray` subclass `GF` intercepts NumPy calls to a
given ufunc, JIT compiles the finite field ufunc (if not already cached), and then invokes the new ufunc on the input array(s).

There are two primary compilation modes: `"jit-lookup"` and `"jit-calculate"`. The supported ufunc compilation modes of a given finite
field are listed in :obj:`~galois.FieldArray.ufunc_modes`.

.. ipython:: python

    GF = galois.GF(3**5)
    GF.ufunc_modes

Large finite fields, which have :obj:`numpy.object_` data types, use `"python-calculate"` which utilizes non-compiled, pure-Python ufuncs.

.. ipython:: python

    GF = galois.GF(2**100)
    GF.ufunc_modes

.. _lookup-tables:

Lookup tables
-------------

The lookup table compilation mode `"jit-lookup"` uses exponential, logarithm, and Zech logarithm lookup tables
to speed up arithmetic computations. These tables are built once at :obj:`~galois.FieldArray` subclass-creation time
during the call to :func:`~galois.GF`.

The exponential and logarithm lookup tables map every finite field element to a power of the primitive element
$\alpha$.

$$x = \alpha^i$$

$$\textrm{log}_{\alpha}(x) = i$$

With these lookup tables, many arithmetic operations are simplified. For instance, multiplication of two finite field
elements is reduced to three lookups and one integer addition.

$$
x \cdot y
&= \alpha^m \cdot \alpha^n \\
&= \alpha^{m + n}
$$

The `Zech logarithm <https://en.wikipedia.org/wiki/Zech%27s_logarithm>`_ is defined below. A similar lookup table is
created for it.

$$1 + \alpha^i = \alpha^{Z(i)}$$

$$Z(i) = \textrm{log}_{\alpha}(1 + \alpha^i)$$

With Zech logarithms, addition of two finite field elements becomes three lookups, one integer addition, and one
integer subtraction.

$$
x + y
&= \alpha^m + \alpha^n \\
&= \alpha^m (1 + \alpha^{n - m}) \\
&= \alpha^m \alpha^{Z(n - m)} \\
&= \alpha^{m + Z(n - m)}
$$

Finite fields with order less than $2^{20}$ use lookup tables by default. In the limited cases where explicit calculation
is faster than table lookup, the explicit calculation is used.

.. ipython:: python

    GF = galois.GF(3**5)
    GF.ufunc_mode

.. _explicit-calculation:

Explicit calculation
--------------------

Finite fields with order greater than $2^{20}$ use explicit calculation by default. This eliminates the need to store large lookup
tables. However, explicit calculation is usually slower than table lookup.

.. ipython:: python

    GF = galois.GF(2**24)
    GF.ufunc_mode

However, if memory is of no concern, even large fields can be compiled to use lookup tables. Initially constructing the lookup tables
may take some time, however.

.. ipython::

    @verbatim
    In [1]: GF = galois.GF(2**24, compile="jit-lookup")

    @verbatim
    In [2]: GF.ufunc_mode
    Out[2]: 'jit-lookup'

Python explicit calculation
---------------------------

Large finite fields cannot use JIT compiled ufuncs. This is because they cannot use NumPy integer data types. This is either
because the order of the field or an intermediate arithmetic result is larger than the max value of :obj:`numpy.int64`.

These finite fields use the :obj:`numpy.object_` data type and have ufunc compilation mode `"python-calculate"`. This mode does *not* compile
the Python functions, but rather converts them into Python ufuncs using :func:`numpy.frompyfunc`. The lack of JIT compilation allows
the ufuncs to operate on Python integers, which have unlimited size. This does come with a performance penalty, however.

.. ipython:: python

    GF = galois.GF(2**100)
    GF.ufunc_mode

Recompile the ufuncs
--------------------

The compilation mode may be explicitly set during creation of the :obj:`~galois.FieldArray` subclass using the
`compile` keyword argument to :func:`~galois.GF`.

Here, the :obj:`~galois.FieldArray` subclass for $\mathrm{GF}(3^5)$ would normally select `"jit-lookup"` as its
default compilation mode. However, we can intentionally choose explicit calculation.

.. ipython:: python

    GF = galois.GF(3**5, compile="jit-calculate")
    GF.ufunc_mode

After a :obj:`~galois.FieldArray` subclass has been created, its compilation mode may be changed using the
:func:`~galois.FieldArray.compile` method.

.. ipython:: python

    GF.compile("jit-lookup")
    GF.ufunc_mode

This will not immediately recompile all of the ufuncs. The ufuncs are compiled on-demand (during their first invocation)
and only if a cached version is not available.
