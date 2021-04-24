Constructing Galois field array classes
=======================================

The main idea of the :obj:`galois` package is that it constructs "Galois field array classes" using `GF = galois.GF(p**m)`.
Galois field array classes, e.g. `GF`, are subclasses of :obj:`numpy.ndarray` and their constructors `a = GF(array_like)` mimic
the :func:`numpy.array` function. Galois field arrays, e.g. `a`, can be operated on like any other numpy array. For example: `a + b`,
`np.reshape(a, new_shape)`, `np.multiply.reduce(a, axis=0)`, etc.

Galois field array classes are subclasses of :obj:`galois.GFArray` with metaclass :obj:`galois.GFMeta`. The metaclass
provides useful methods and attributes related to the finite field.

The Galois field :math:`\mathrm{GF}(2)` is already constructed in :obj:`galois`. It can be accessed by :obj:`galois.GF2`.

.. ipython:: python

   GF2 = galois.GF2
   print(GF2)
   issubclass(GF2, np.ndarray)
   issubclass(GF2, galois.GFArray)
   issubclass(type(GF2), galois.GFMeta)
   print(GF2.properties)

:math:`\mathrm{GF}(2^m)` fields, where :math:`m` is a positive integer, can be constructed using the class
factory :func:`galois.GF`.

.. ipython:: python

   GF8 = galois.GF(2**3)
   print(GF8)
   issubclass(GF8, np.ndarray)
   issubclass(GF8, galois.GFArray)
   issubclass(type(GF8), galois.GFMeta)
   print(GF8.properties)

:math:`\mathrm{GF}(p)` fields, where :math:`p` is prime, can be constructed using the class factory
:func:`galois.GF`.

.. ipython:: python

   GF7 = galois.GF(7)
   print(GF7)
   issubclass(GF7, np.ndarray)
   issubclass(GF7, galois.GFArray)
   issubclass(type(GF7), galois.GFMeta)
   print(GF7.properties)
