Extremely large fields
======================

Arbitrarily-large :math:`\mathrm{GF}(2^m)`, :math:`\mathrm{GF}(p)`, :math:`\mathrm{GF}(p^m)` fields are supported.
Because field elements can't be represented with :obj:`numpy.int64`, we use `dtype=object` in the `numpy` arrays. This enables
use of native python :obj:`int`, which doesn't overflow. It comes at a performance cost though. There are no JIT-compiled
arithmetic ufuncs. All the arithmetic is done in pure python. All the same array operations, broadcasting, ufunc methods,
etc are supported.


Large GF(p) fields
------------------

.. ipython:: python

   prime = 36893488147419103183
   galois.is_prime(prime)

   GF = galois.GF(prime)
   print(GF)

   a = GF.Random(10); a
   b = GF.Random(10); b

   a + b


Large GF(2^m) fields
--------------------

.. ipython:: python

   GF = galois.GF(2**100)
   print(GF)

   a = GF([2**8, 2**21, 2**35, 2**98]); a
   b = GF([2**91, 2**40, 2**40, 2**2]); b

   a + b

   # Display elements as polynomials
   GF.display("poly")

   a
   b
   a + b
   a * b

   # Reset the display mode
   GF.display()
