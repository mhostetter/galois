.. image:: ../logo/galois-heading.png
   :align: center

The :obj:`galois` library is a Python 3 package that extends NumPy arrays to operate over finite fields.

The user creates a :obj:`~galois.FieldArray` subclass using `GF = galois.GF(p**m)`. `GF` is a subclass of :obj:`numpy.ndarray`
and its constructor `x = GF(array_like)` mimics the signature of :func:`numpy.array`. The :obj:`~galois.FieldArray` `x` is operated
on like any other NumPy array except all arithmetic is performed in :math:`\mathrm{GF}(p^m)`, not :math:`\mathbb{R}`.

Internally, the finite field arithmetic is implemented by replacing `NumPy ufuncs <https://numpy.org/doc/stable/reference/ufuncs.html>`_.
The new ufuncs are written in pure Python and `just-in-time compiled <https://numba.pydata.org/numba-doc/dev/user/vectorize.html>`_ with
`Numba <https://numba.pydata.org/>`_. The ufuncs can be configured to use either lookup tables (for speed) or explicit
calculation (for memory savings).

.. admonition:: Disclaimer
   :class: warning

   The algorithms implemented in the NumPy ufuncs are not constant-time, but were instead designed for performance. As such, the
   library could be vulnerable to a `side-channel timing attack <https://en.wikipedia.org/wiki/Timing_attack>`_. This library is not
   intended for production security, but instead for research & development, reverse engineering, cryptanalysis, experimentation,
   and general education.

Features
--------

- Supports all Galois fields :math:`\mathrm{GF}(p^m)`, even arbitrarily-large fields!
- **Faster** than native NumPy! `GF(x) * GF(y)` is faster than `(x * y) % p` for :math:`\mathrm{GF}(p)`.
- Seamless integration with NumPy -- normal NumPy functions work on :obj:`~galois.FieldArray` instances.
- Linear algebra over finite fields using normal :obj:`numpy.linalg` functions.
- Linear transforms over finite fields, such as the FFT with :func:`numpy.fft.fft` and the NTT with :func:`~galois.ntt`.
- Functions to generate irreducible, primitive, and Conway polynomials.
- Univariate polynomials over finite fields with :obj:`~galois.Poly`.
- Forward error correction codes with :obj:`~galois.BCH` and :obj:`~galois.ReedSolomon`.
- Fibonacci and Galois linear-feedback shift registers over any finite field with :obj:`~galois.FLFSR` and :obj:`~galois.GLFSR`.
- Various number theoretic functions.
- Integer factorization and accompanying algorithms.
- Prime number generation and primality testing.

Roadmap
-------

- Elliptic curves over finite fields
- Galois ring arrays
- GPU support

Acknowledgements
----------------

The :obj:`galois` library is an extension of, and completely dependent on, `NumPy <https://numpy.org/>`_. It also heavily
relies on `Numba <https://numba.pydata.org/>`_ and the `LLVM just-in-time compiler <https://llvm.org/>`_ for optimizing performance
of the finite field arithmetic.

`Frank Luebeck's compilation <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_ of Conway polynomials and
`Wolfram's compilation <https://datarepository.wolframcloud.com/resources/Primitive-Polynomials/>`_ of primitive polynomials are used
for efficient polynomial lookup, when possible.

`Sage <https://www.sagemath.org/>`_ is used extensively for generating test vectors for finite field arithmetic and polynomial arithmetic.
`SymPy <https://www.sympy.org/en/index.html>`_ is used to generate some test vectors. `Octave <https://www.gnu.org/software/octave/index>`_
is used to generate test vectors for forward error correction codes.

This library would not be possible without all of the other libraries mentioned. Thank you to all their developers!

Citation
--------

If this library was useful to you in your research, please cite us. Following the `GitHub citation standards <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files>`_, here is the recommended citation.

.. md-tab-set::

   .. md-tab-item:: BibTeX

      .. code-block:: latex

         @software{Hostetter_Galois_2020,
            title = {{Galois: A performant NumPy extension for Galois fields}},
            author = {Hostetter, Matt},
            month = {11},
            year = {2020},
            url = {https://github.com/mhostetter/galois},
         }

   .. md-tab-item:: APA

      .. code-block:: text

         Hostetter, M. (2020). Galois: A performant NumPy extension for Galois fields [Computer software]. https://github.com/mhostetter/galois


.. toctree::
   :caption: Getting Started
   :hidden:

   getting-started.rst

.. toctree::
   :caption: Basic Usage
   :hidden:

   basic-usage/array-classes.rst
   basic-usage/compilation-modes.rst
   basic-usage/element-representation.rst
   basic-usage/array-creation.rst
   basic-usage/array-arithmetic.rst
   basic-usage/poly.rst
   basic-usage/poly-arithmetic.rst

.. toctree::
   :caption: Tutorials
   :hidden:

   tutorials/intro-to-prime-fields.rst
   tutorials/intro-to-extension-fields.rst

.. toctree::
   :caption: Performance
   :hidden:

   performance/prime-fields.rst
   performance/binary-extension-fields.rst
   performance/benchmarks.rst

.. toctree::
   :caption: Development
   :hidden:

   development/installation.rst
   development/linter.rst
   development/unit-tests.rst
   development/documentation.rst

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api.rst

.. toctree::
   :caption: Release Notes
   :hidden:

   release-notes/versioning.rst
   release-notes/v0.2.0.md
   release-notes/v0.1.2.md
   release-notes/v0.1.1.md
   release-notes/v0.1.0.md
   release-notes/v0.0.33.md
   release-notes/v0.0.32.md
   release-notes/v0.0.31.md
   release-notes/v0.0.30.md
   release-notes/v0.0.29.md
   release-notes/v0.0.28.md
   release-notes/v0.0.27.md
   release-notes/v0.0.26.md
   release-notes/v0.0.25.md
   release-notes/v0.0.24.md
   release-notes/v0.0.23.md
   release-notes/v0.0.22.md
   release-notes/v0.0.21.md
   release-notes/v0.0.20.md
   release-notes/v0.0.19.md
   release-notes/v0.0.18.md
   release-notes/v0.0.17.md
   release-notes/v0.0.16.md
   release-notes/v0.0.15.md
   release-notes/v0.0.14.md

.. toctree::
   :caption: Index
   :hidden:

   genindex
