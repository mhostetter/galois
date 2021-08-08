Features
========

- Supports all Galois fields :math:`\mathrm{GF}(p^m)`, even arbitrarily-large fields!
- **Faster** than native NumPy! `GF(x) * GF(y)` is faster than `(x * y) % p` for :math:`\mathrm{GF}(p)`
- Seamless integration with NumPy -- normal NumPy functions work on Galois field arrays
- Linear algebra on Galois field matrices using normal `np.linalg` functions
- Functions to generate irreducible, primitive, and Conway polynomials
- Polynomials over Galois fields with :obj:`galois.Poly`
- Forward error correction codes with :obj:`galois.BCH` and :obj:`galois.ReedSolomon`
- Fibonacci and Galois linear feedback shift registers with :obj:`galois.LFSR`, both binary and p-ary
- Various number theoretic functions
- Integer factorization and accompanying algorithms
- Prime number generation and primality testing
