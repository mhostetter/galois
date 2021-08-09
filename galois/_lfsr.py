"""
A module containing classes and functions for generating and analyzing linear feedback shift registers and their sequences.
"""
import numba
from numba import int64
import numpy as np

from ._field import FieldArray, Poly
from ._field._meta_function import UNARY_CALCULATE_SIG, BINARY_CALCULATE_SIG
from ._field._poly_conversion import integer_to_poly
from ._overrides import set_module

__all__ = ["LFSR", "berlekamp_massey"]


@set_module("galois")
class LFSR:
    r"""
    Implements a linear-feedback shift register (LFSR).

    This class implements an LFSR in either the Fibonacci or Galois configuration. An LFSR is defined
    by its generator polynomial :math:`g(x) = g_n x^n + \dots + g_1 x + g_0` and initial state vector
    :math:`s = [s_{n-1}, \dots, s_1, s_0]`.

    Parameters
    ----------
    poly : galois.Poly
        The generator polynomial :math:`g(x) = g_n x^n + \dots + g_1 x + g_0`.
    state : int, tuple, list, numpy.ndarray, galois.FieldArray, optional
        The initial state vector :math:`s = [s_{n-1}, \dots, s_1, s_0]`. If specified as an integer, then
        :math:`s_{n-1}` is interpreted as the MSB and :math:`s_0` as the LSB. The default is 1 which corresponds to
        :math:`s = [0, \dots, 0, 1]`.
    config : str, optional
        A string indicating the LFSR feedback configuration, either `"fibonacci"` (default) or `"galois"`.

    Notes
    -----
    Below are diagrams for a degree-:math:`3` LFSR in the Fibonacci and Galois configuration. The generator
    polynomial is :math:`g(x) = g_3x^3 + g_2x^2 + g_1x + g_0` and state vector is :math:`s = [s_2, s_1, s_0]`.

    .. code-block:: text
       :caption: Fibonacci LFSR Configuration

             ┌────────────⊕◀───────────⊕◀───────────┐
             │            ▲            ▲            |
        g0 ⊗─┤       g1 ⊗─┤       g2 ⊗─┤       g3 ⊗─┤
             │  ┏━━━━━━┓  |  ┏━━━━━━┓  |  ┏━━━━━━┓  |
             └─▶┃  s2  ┃──┴─▶┃  s1  ┃──┴─▶┃  s0  ┃──┴──▶ y[n]
                ┗━━━━━━┛     ┗━━━━━━┛     ┗━━━━━━┛

    In the Fibonacci configuration, at time instant :math:`i` the next :math:`n-1` outputs are the current state reversed, that is :math:`[y_i, y_{i+1}, \dots, y_{i+n-1}] = [s_0, s_1, \dots, s_{n-1}]`.
    And the :math:`n`-th output is a linear combination of the current state and the generator polynomial :math:`y_{i+n} = (g_n s_0 + g_{n-1} s_1 + \dots + g_1 s_{n-1}) g_0`.

    .. code-block:: text
       :caption: Galois LFSR Configuration

             ┌────────────┬────────────┬────────────┐
        g0 ⊗─┤       g1 ⊗─┤       g2 ⊗─┤       g3 ⊗─┤
             │            |            |            |
             │  ┏━━━━━━┓  ▼  ┏━━━━━━┓  ▼  ┏━━━━━━┓  |
             └─▶┃  s0  ┃──⊕─▶┃  s1  ┃──⊕─▶┃  s2  ┃──┴──▶ y[n]
                ┗━━━━━━┛     ┗━━━━━━┛     ┗━━━━━━┛

    In the Galois configuration, the next output is :math:`y = s_{n-1}` and the next state is computed by :math:`s_k = s_{n-1} g_n g_k + s_{k-1}`. In the case of
    :math:`s_0` there is no previous state added.
    """

    def __init__(self, poly, state=1, config="fibonacci"):
        if not isinstance(poly, Poly):
            raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
        if not isinstance(state, (type(None), int, np.integer, tuple, list, np.ndarray, FieldArray)):
            raise TypeError(f"Argument `state` must be an int or array-like, not {type(state)}.")
        if not isinstance(config, str):
            raise TypeError(f"Argument `config` must be a string, not {type(config)}.")
        if not config in ["fibonacci", "galois"]:
            raise ValueError(f"Argument `config` must be in ['fibonacci', 'galois'], not {state!r}.")

        # Convert integer state to vector state
        if isinstance(state, (int, np.integer)):
            state = integer_to_poly(state, poly.field.order, degree=poly.degree - 1)

        self._field = poly.field
        self._poly = poly
        self._initial_state = self.field(state)
        self._state = None
        self._config = config
        self.reset()

        # Pre-compile the arithmetic functions and JIT routines
        if self.field._ufunc_mode != "python-calculate":
            self._add = self.field._calculate_jit("add")
            self._multiply = self.field._calculate_jit("multiply")
            self._step = jit_calculate(f"{self.config}_lfsr_step")
        else:
            self._add = self.field._python_func("add")
            self._multiply = self.field._python_func("multiply")
            self._step = python_func(f"{self.config}_lfsr_step")

    def __str__(self):
        if self.config == "fibonacci":
            return f"<Fibonacci LFSR: poly={self.poly}>"
        else:
            return f"<Galois LFSR: poly={self.poly}>"

    def __repr__(self):
        return str(self)

    def reset(self):
        """
        Resets the LFSR state to the initial state.

        Examples
        --------
        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4)); lfsr
            lfsr.state
            lfsr.step(10)
            lfsr.state
            lfsr.reset()
            lfsr.state
        """
        self._state = self.initial_state.copy()

    def step(self, steps=1):
        """
        Steps the LFSR and produces `steps` output symbols.

        Parameters
        ----------
        steps : int, optional
            The number of output symbols to produce. The default is 1.

        Returns
        -------
        galois.FieldArray
            An array of output symbols of type :obj:`field` with size `steps`.

        Examples
        --------
        Step the LFSR one output at a time.

        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4)); lfsr
            lfsr.state
            lfsr.state, lfsr.step()
            lfsr.state, lfsr.step()
            lfsr.state, lfsr.step()
            lfsr.state, lfsr.step()

        Step the LFSR 10 steps.

        .. ipython:: python

            lfsr.reset()
            lfsr.step(10)
        """
        if not isinstance(steps, (int, np.integer)):
            raise TypeError(f"Argument `steps` must be an integer, not {type(steps)}.")
        if not steps >= 1:
            raise ValueError(f"Argument `steps` must be at least 1, not {steps}.")

        if self.field.ufunc_mode != "python-calculate":
            poly = self.poly.coeffs.astype(np.int64)
            state = self.state.astype(np.int64)
            y = self._step(poly, state, steps, self._add, self._multiply, self.field.characteristic, self.field.degree, self.field._irreducible_poly_int)
            y = y.astype(self.state.dtype)
        else:
            poly = self.poly.coeffs.view(np.ndarray)
            state = self.state.view(np.ndarray)
            y = self._step(poly, state, steps, self._add, self._multiply, self.field.characteristic, self.field.degree, self.field._irreducible_poly_int)

        self._state[:] = state[:]
        y = y.view(self.field)
        if y.size == 1:
            y = y[0]

        return y

    @property
    def field(self):
        """
        galois.FieldClass: The Galois field that defines the LFSR arithmetic. The generator polynomial :math:`g(x)` is over this
        field and the state vector contains values in this field.

        Examples
        --------
        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4)); lfsr
            lfsr.field
            print(lfsr.field.properties)
        """
        return self._field

    @property
    def poly(self):
        r"""
        galois.Poly: The generator polynomial :math:`g(x) = g_n x^n + \dots + g_1 x + g_0`.

        Examples
        --------
        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4)); lfsr
            lfsr.poly
        """
        return self._poly

    @property
    def initial_state(self):
        r"""
        galois.FieldArray: The initial state vector :math:`s = [s_{n-1}, \dots, s_1, s_0]`.

        Examples
        --------
        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4)); lfsr
            lfsr.initial_state
        """
        return self._initial_state.copy()

    @property
    def state(self):
        r"""
        galois.FieldArray: The current state vector :math:`s = [s_{n-1}, \dots, s_1, s_0]`.

        Examples
        --------
        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4)); lfsr
            lfsr.state
            lfsr.step(10)
            lfsr.state
        """
        return self._state.copy()

    @property
    def config(self):
        """
        str: The LFSR configuration, either `"fibonacci"` or `"galois"`. See the Notes section of :obj:`LFSR` for descriptions of
        the two configurations.

        Examples
        --------
        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4)); lfsr
            lfsr.config

        .. ipython:: python

            lfsr = galois.LFSR(galois.primitive_poly(2, 4), config="galois"); lfsr
            lfsr.config
        """
        return self._config


# @set_module("galois")
# def pary_lfsr_poly(poly):
#     field = poly.field
#     p = field.characteristic
#     coeffs = poly.coeffs
#     c = field(p - 1) / coeffs[-1]
#     coeffs[0:-1] *= c
#     # coeffs[-1] = field(1)
#     # c = field(p - 1) / coeffs[0]
#     # coeffs[1:] *= c
#     # coeffs[0] = field(p - 1)
#     return Poly(coeffs, field=field)


@set_module("galois")
def berlekamp_massey(sequence, config="fibonacci", state=False):
    r"""
    Finds the minimum-degree polynomial :math:`c(x)` that produces the sequence in :math:`\mathrm{GF}(p^m)`.

    This function implements the Berlekamp-Massey algorithm.

    Parameters
    ----------
    sequence : galois.FieldArray
        A 1-D sequence of Galois field elements in :math:`\mathrm{GF}(p^m)`.
    config : str, optional
        A string indicating the LFSR feedback configuration for the returned connection polynomial, either `"fibonacci"`
        (default) or `"galois"`. See the LFSR configurations in :obj:`galois.LFSR`. The LFSR configuration will indicate
        if the connection polynomial coefficients should be reversed or not.
    state : bool, optional
        Indicates whether to return the LFSR initial state such that the output is the input sequence. The default is `False`.

    Returns
    -------
    galois.Poly
        The minimum-degree polynomial :math:`c(x) \in \mathrm{GF}(p^m)[x]` that produces the input sequence.
    galois.FieldArray
        The initial state of the LFSR such that the output will generate the input sequence. Only returned if `state=True`.

    References
    ----------
    * https://crypto.stanford.edu/~mironov/cs359/massey.pdf
    * https://www.embeddedrelated.com/showarticle/1099.php

    Examples
    --------
    The Berlekamp-Massey algorithm requires :math:`2n` output symbols to determine the :math:`n`-degree minimum connection
    polynomial.

    .. ipython:: python

        g = galois.conway_poly(2, 8); g
        lfsr = galois.LFSR(g, state=1); lfsr
        s = lfsr.step(16); s
        galois.berlekamp_massey(s)
    """
    if not isinstance(sequence, FieldArray):
        raise TypeError(f"Argument `sequence` must be a Galois field array, not {type(sequence)}.")
    if not isinstance(config, str):
        raise TypeError(f"Argument `config` must be a string, not {type(config)}.")
    if not isinstance(state, bool):
        raise TypeError(f"Argument `state` must be a bool, not {type(state)}.")
    if not sequence.ndim == 1:
        raise ValueError(f"Argument `sequence` must be 1-D, not {sequence.ndim}-D.")
    if not config in ["fibonacci", "galois"]:
        raise ValueError(f"Argument `config` must be in ['fibonacci', 'galois'], not {state!r}.")

    field = type(sequence)
    dtype = sequence.dtype

    if field.ufunc_mode != "python-calculate":
        sequence = sequence.astype(np.int64)
        add = field._calculate_jit("add")
        subtract = field._calculate_jit("subtract")
        multiply = field._calculate_jit("multiply")
        reciprocal = field._calculate_jit("reciprocal")
        coeffs = jit_calculate("berlekamp_massey")(sequence, add, subtract, multiply, reciprocal, field.characteristic, field.degree, field._irreducible_poly_int)
        coeffs = coeffs.astype(dtype)
    else:
        sequence = sequence.view(np.ndarray)
        subtract = field._python_func("subtract")
        multiply = field._python_func("multiply")
        divide = field._python_func("divide")
        reciprocal = field._python_func("reciprocal")
        coeffs = python_func("berlekamp_massey")(sequence, subtract, multiply, divide, reciprocal, field.characteristic, field.degree, field._irreducible_poly_int)

    if config == "fibonacci":
        poly = Poly(coeffs, field=field)
        state_ = sequence[0:poly.degree][::-1]
    else:
        poly = Poly(coeffs[::-1], field=field)
        state_ = sequence[0:poly.degree][::-1]

    if not state:
        return poly
    else:
        return poly, state_


###############################################################################
# JIT functions
###############################################################################

def jit_calculate(name):
    if name not in jit_calculate.cache:
        function = eval(f"{name}_calculate")
        sig = eval(f"{name.upper()}_CALCULATE_SIG")
        jit_calculate.cache[name] = numba.jit(sig.signature, nopython=True, cache=True)(function)
    return jit_calculate.cache[name]

jit_calculate.cache = {}


def python_func(name):
    return eval(f"{name}_calculate")


FIBONACCI_LFSR_STEP_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], int64, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, int64, int64, int64))

def fibonacci_lfsr_step_calculate(poly, state, steps, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
    dtype = state.dtype

    poly = poly[::-1]  # Place the MSB on the right
    y = np.zeros(steps, dtype=dtype)

    for i in range(steps):
        # Dot product state with generator polynomial
        s = 0
        for j in range(state.size):
            s = ADD(s, MULTIPLY(state[j], poly[1 + j], *args), *args)

        s = MULTIPLY(s, poly[0], *args)  # Scale computed value by LSB polynomial coefficient
        y[i] = state[-1]  # Output is popped off the shift register
        state[1:] = state[0:-1]  # Shift state rightward, from MSB to LSB
        state[0] = s  # Insert computed value at leftmost position

    return y


GALOIS_LFSR_STEP_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], int64, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, int64, int64, int64))

def galois_lfsr_step_calculate(poly, state, steps, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
    dtype = state.dtype

    poly = poly[::-1]  # Place the MSB on the right
    state = state[::-1]  # Place the MSB on the right
    y = np.zeros(steps, dtype=dtype)

    for i in range(steps):
        y[i] = state[-1]  # Output is popped off the shift register
        s = MULTIPLY(y[i], poly[0], *args)  # Scale output value by MSB polynomial coefficient

        state[1:] = state[0:-1]  # Shift state rightward, from LSB to MSB
        state[0] = 0  # Set leftmost state to 0

        if s > 0:
            # Inline add feedback value multiplied by polynomial coefficient
            for j in range(state.size):
                state[j] = ADD(state[j], MULTIPLY(s, poly[j], *args), *args)

    state = state[::-1]  # Place the MSB on the left before returning

    return y


BERLEKAMP_MASSEY_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, BINARY_CALCULATE_SIG, UNARY_CALCULATE_SIG, int64, int64, int64))

def berlekamp_massey_calculate(sequence, ADD, SUBTRACT, MULTIPLY, RECIPROCAL, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
    args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
    dtype = sequence.dtype

    N = sequence.size

    s = sequence
    c = np.zeros(N, dtype=dtype)
    b = np.zeros(N, dtype=dtype)
    c[0] = 1  # The polynomial c(x) = 1
    b[0] = 1  # The polynomial b(x) = 1
    L = 0
    m = 1
    bb = 1

    for n in range(0, N):
        d = 0
        for i in range(0, L + 1):
            d = ADD(d, MULTIPLY(s[n - i], c[i], *args), *args)

        if d == 0:
            m += 1
        elif 2*L <= n:
            t = c.copy()
            d_bb = MULTIPLY(d, RECIPROCAL(bb, *args), *args)
            for i in range(m, N):
                c[i] = SUBTRACT(c[i], MULTIPLY(d_bb, b[i - m], *args), *args)
            L = n + 1 - L
            b = t.copy()
            bb = d
            m = 1
        else:
            d_bb = MULTIPLY(d, RECIPROCAL(bb, *args), *args)
            for i in range(m, N):
                c[i] = SUBTRACT(c[i], MULTIPLY(d_bb, b[i - m], *args), *args)
            m += 1

    # Return the coefficients for a Fibonacci LFSR
    return c[0:L + 1][::-1]
