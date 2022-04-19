Transforms
==========

This section contains classes and functions for various transforms using Galois fields.

.. currentmodule:: galois

Number-theoretic transform
--------------------------

.. autosummary::
   :toctree:

   ntt
   intt

Discrete Fourier transform
--------------------------

The DFT over arbitrary finite fields may be performed by invoking :func:`numpy.fft.fft` on a :obj:`FieldArray`. The same is
true for the inverse DFT and :func:`numpy.fft.ifft`. See :ref:`advanced arithmetic`.
