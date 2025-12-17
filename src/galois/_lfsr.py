"""
A module containing classes and functions for generating and analyzing linear feedback shift registers and their
sequences.
"""

from __future__ import annotations

from typing import Type, overload

import numba
import numpy as np
from numba import int64
from typing_extensions import Literal, Self

from ._domains._function import Function
from ._fields import FieldArray
from ._helper import export, verify_isinstance
from ._options import printoptions
from ._polys import Poly
from .typing import ArrayLike

###############################################################################
# LFSR base class
###############################################################################


class _LFSR:
    r"""
    A linear-feedback shift register base object.
    """

    _type = "fibonacci"

    def __init__(
        self,
        feedback_poly: Poly,
        state: ArrayLike | None = None,
    ):
        verify_isinstance(feedback_poly, Poly)
        if not feedback_poly.coeffs[-1] == 1:
            raise ValueError(f"Argument 'feedback_poly' must have a 0-th degree term of 1, not {feedback_poly}.")

        self._field = feedback_poly.field
        self._feedback_poly = feedback_poly
        self._characteristic_poly = feedback_poly.reverse()
        self._order = feedback_poly.degree

        if self._type == "fibonacci":
            # T = [-a_1, -a_2, ..., -a_n]
            # c(x) = x^{n} + a_{1}x^{n-1} + a_{2}x^{n-2} + \dots + a_{n}
            self._taps = -self.characteristic_poly.coeffs[1:]
        else:
            # T = [-a_n, -a_{n-1}, ..., -a_2, -a_1]
            # c(x) = x^{n} + a_{1}x^{n-1} + a_{2}x^{n-2} + \dots + a_{n}
            self._taps = -self.characteristic_poly.coeffs[1:][::-1]

        if state is None:
            state = self.field.Ones(self.order)

        self._initial_state = self._verify_and_convert_state(state)
        self._state = self.initial_state.copy()

    @classmethod
    def Taps(cls, taps: FieldArray, state: ArrayLike | None = None) -> Self:
        verify_isinstance(taps, FieldArray)

        if cls._type == "fibonacci":
            # T = [-a_1, -a_2, ..., -a_n]
            # f(x) = 1 + a_{1}x + a_{2}x^{2} + \dots + a_{n}x^{n}
            coeffs = -taps
            coeffs = np.append(1, coeffs)  # Add x^0 term
            feedback_poly = Poly(coeffs[::-1])  # Make degree descending
        else:
            # T = [-a_n, -a_{n-1}, ..., -a_2, -a_1]
            # c(x) = x^{n} + a_{1}x^{n-1} + a_{2}x^{n-2} + \dots + a_{n}
            # f(x) = 1 + a_{1}x + a_{2}x^{2} + \dots + a_{n}x^{n}
            coeffs = -taps
            coeffs = np.append(1, coeffs)  # Add x^n term
            characteristic_poly = Poly(coeffs)
            feedback_poly = characteristic_poly.reverse()

        return cls(feedback_poly, state=state)

    def _verify_and_convert_state(self, state: ArrayLike):
        verify_isinstance(state, (tuple, list, np.ndarray, FieldArray))

        state = self.field(state)  # Coerce array-like object to field array

        if not state.size == self.order:
            raise ValueError(
                f"Argument 'state' must have size equal to the degree of the characteristic polynomial, "
                f"not {state.size} and {self.characteristic_poly.degree}."
            )

        return state

    def reset(self, state: ArrayLike | None = None):
        state = self.initial_state if state is None else state
        self._state = self._verify_and_convert_state(state)

    def step(self, steps: int = 1) -> FieldArray:
        verify_isinstance(steps, int)

        if steps == 0:
            y = self.field([])
        elif steps > 0:
            y = self._step_forward(steps)
        else:
            y = self._step_backward(abs(steps))

        return y

    def _step_forward(self, steps):
        assert steps > 0

        if self._type == "fibonacci":
            y, state = fibonacci_lfsr_step_forward_jit(self.field)(self.taps, self.state, steps)
        else:
            y, state = galois_lfsr_step_forward_jit(self.field)(self.taps, self.state, steps)

        self._state[:] = state[:]
        if y.size == 1:
            y = y[0]

        return y

    def _step_backward(self, steps):
        assert steps > 0

        if self.characteristic_poly.coeffs[-1] == 0:
            raise ValueError(
                "Can only step the shift register backwards if the a_n tap is non-zero, "
                f"not c(x) = {self.characteristic_poly}."
            )

        if self._type == "fibonacci":
            y, state = fibonacci_lfsr_step_backward_jit(self.field)(self.taps, self.state, steps)
        else:
            y, state = galois_lfsr_step_backward_jit(self.field)(self.taps, self.state, steps)

        self._state[:] = state[:]
        if y.size == 1:
            y = y[0]

        return y

    @property
    def field(self) -> Type[FieldArray]:
        return self._field

    @property
    def feedback_poly(self) -> Poly:
        return self._feedback_poly

    @property
    def characteristic_poly(self) -> Poly:
        return self._characteristic_poly

    @property
    def taps(self) -> FieldArray:
        return self._taps

    @property
    def order(self) -> int:
        return self._order

    @property
    def initial_state(self) -> FieldArray:
        return self._initial_state.copy()

    @property
    def state(self) -> FieldArray:
        return self._state.copy()


###############################################################################
# Fibonacci LFSR
###############################################################################


@export
class FLFSR(_LFSR):
    r"""
    A Fibonacci linear-feedback shift register (LFSR).

    Notes:
        A Fibonacci LFSR is defined by its feedback (connection) polynomial

        $$
        f(x) = 1 + a_1 x + a_2 x^2 + \dots + a_{n} x^{n},
        $$

        where $f(0) = 1$ and the degree $n$ equals the length of the shift register. The associated output sequence
        $y[t]$ satisfies the linear recurrence

        $$
        y[t] + a_1 y[t-1] + a_2 y[t-2] + \dots + a_{n} y[t-n] = 0.
        $$

        The characteristic polynomial of the sequence is the reciprocal of the feedback polynomial

        $$
        c(x) &= x^{n} + a_1 x^{n-1} + a_2 x^{n-2} + \dots + a_{n} \\
        &= x^{n} f(x^{-1}).
        $$

        In the Fibonacci configuration, the shift register is arranged so that its taps implement the recurrence
        directly. The taps are simply the feedback coefficients $[-a_1, -a_2, \dots, -a_n]$ in a fixed left-to-right
        order that matches the chosen shift direction.

        .. code-block:: text
            :caption: Fibonacci LFSR Configuration

               y[t]
                +--------------+<-------------+<-------------+<-------------+
                |              ^              ^              ^              |
                |              | -a_1         | -a_2         | -a_{n-1}     | -a_n
                |              | T[0]         | T[1]         | T[n-2]       | T[n-1]
                |  +--------+  |  +--------+  |              |  +--------+  |
                +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t-n]
                   +--------+     +--------+                    +--------+
                     y[t-1]         y[t-2]                        y[t-n]

        The state vector is stored left-to-right as $S = [S_0, S_1, \dots, S_{n-1}]$.

    References:
        - Gardner, D. 2019. “Applications of the Galois Model LFSR in Cryptography”. figshare.
          https://hdl.handle.net/2134/21932.

    See Also:
        berlekamp_massey

    Examples:
        .. md-tab-set::

            .. md-tab-item:: GF(2)

                Create a Fibonacci LFSR from a degree-4 primitive characteristic polynomial over
                $\mathrm{GF}(2)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(2, 4).reverse(); feedback_poly
                    lfsr = galois.FLFSR(feedback_poly)
                    print(lfsr)

                Step the Fibonacci LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

            .. md-tab-item:: GF(p)

                Create a Fibonacci LFSR from a degree-4 primitive characteristic polynomial over
                $\mathrm{GF}(7)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                    lfsr = galois.FLFSR(feedback_poly)
                    print(lfsr)

                Step the Fibonacci LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

            .. md-tab-item:: GF(2^m)

                Create a Fibonacci LFSR from a degree-4 primitive characteristic polynomial over
                $\mathrm{GF}(2^3)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(2**3, 4).reverse(); feedback_poly
                    lfsr = galois.FLFSR(feedback_poly)
                    print(lfsr)

                Step the Fibonacci LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

            .. md-tab-item:: GF(p^m)

                Create a Fibonacci LFSR from a degree-4 primitive characteristic polynomial over
                $\mathrm{GF}(3^3)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(3**3, 4).reverse(); feedback_poly
                    lfsr = galois.FLFSR(feedback_poly)
                    print(lfsr)

                Step the Fibonacci LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

    Group:
        linear-sequences
    """

    _type = "fibonacci"

    def __init__(
        self,
        feedback_poly: Poly,
        state: ArrayLike | None = None,
    ):
        r"""
        Constructs a Fibonacci LFSR from its feedback polynomial $f(x)$.

        Arguments:
            feedback_poly: The feedback polynomial $f(x) = 1 + a_1 x + a_2 x^2 + \dots + a_{n} x^{n}$.
            state: The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None`
                which corresponds to all ones.

        See Also:
            irreducible_poly, primitive_poly

        Notes:
            A Fibonacci LFSR may be constructed from its characteristic polynomial $c(x)$ by passing in its
            reciprocal as the feedback polynomial. This is because $f(x) = x^n c(x^{-1})$.
        """
        super().__init__(feedback_poly, state=state)

    @classmethod
    def Taps(cls, taps: FieldArray, state: ArrayLike | None = None) -> Self:
        r"""
        Constructs a Fibonacci LFSR from its taps $T = [-a_1, -a_2, \dots, -a_n]$.

        Arguments:
            taps: The shift register taps $T = [-a_1, -a_2, \dots, -a_n]$.
            state: The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None`
                which corresponds to all ones.

        Returns:
            A Fibonacci LFSR with taps $T = [-a_1, -a_2, \dots, -a_n]$.

        Examples:
            .. ipython:: python

                characteristic_poly = galois.primitive_poly(7, 4); characteristic_poly
                taps = -characteristic_poly.coeffs[1:]; taps
                lfsr = galois.FLFSR.Taps(taps)
                print(lfsr)
        """
        return super().Taps(taps, state=state)

    def __repr__(self) -> str:
        """
        A terse representation of the Fibonacci LFSR.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly)
                lfsr
        """
        with printoptions(coeffs="asc"):
            return f"<Fibonacci LFSR: f(x) = {self.feedback_poly} over {self.field.name}>"

    def __str__(self) -> str:
        """
        A formatted string of relevant properties of the Fibonacci LFSR.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly)
                print(lfsr)
        """
        string = "Fibonacci LFSR:"
        string += f"\n  field: {self.field.name}"
        with printoptions(coeffs="asc"):
            string += f"\n  feedback_poly: {self.feedback_poly}"
        string += f"\n  characteristic_poly: {self.characteristic_poly}"
        string += f"\n  taps: {self.taps}"
        string += f"\n  order: {self.order}"
        string += f"\n  state: {self.state}"
        string += f"\n  initial_state: {self.initial_state}"

        return string

    def reset(self, state: ArrayLike | None = None):
        r"""
        Resets the Fibonacci LFSR state to the specified state.

        Arguments:
            state: The state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None` which
                corresponds to the initial state.

        Examples:
            Step the Fibonacci LFSR 10 steps to modify its state.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly); lfsr
                lfsr.state
                lfsr.step(10)
                lfsr.state

            Reset the Fibonacci LFSR state.

            .. ipython:: python

                lfsr.reset()
                lfsr.state

            Create an Fibonacci LFSR and view its initial state.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly); lfsr
                lfsr.state

            Reset the Fibonacci LFSR state to a new state.

            .. ipython:: python

                lfsr.reset([1, 2, 3, 4])
                lfsr.state
        """
        return super().reset(state)

    def step(self, steps: int = 1) -> FieldArray:
        """
        Produces the next `steps` output symbols.

        Arguments:
            steps: The direction and number of output symbols to produce. The default is 1. If negative, the
                Fibonacci LFSR will step backwards.

        Returns:
            An array of output symbols of type :obj:`field` with size `abs(steps)`.

        Examples:
            Step the Fibonacci LFSR one output at a time. Notice the first $n$ outputs of a Fibonacci LFSR are
            its state reversed.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                # Ending state
                lfsr.state

            Step the Fibonacci LFSR 5 steps in one call. This is more efficient than iterating one output at a time.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.state
                lfsr.step(5)
                # Ending state
                lfsr.state

            Step the Fibonacci LFSR 5 steps backward. Notice the output sequence is the reverse of the original
            sequence. Also notice the ending state is the same as the initial state.

            .. ipython:: python

                lfsr.step(-5)
                lfsr.state
        """
        return super().step(steps)

    def to_galois_lfsr(self) -> GLFSR:
        r"""
        Converts the Fibonacci LFSR to a Galois LFSR that produces the same output sequence.

        Returns:
            An equivalent Galois LFSR.

        Notes:
            Let $Y(x)$ be the polynomial formed from the next $n$ outputs of the Fibonacci LFSR,
            where $n$ is the order.

            $$Y(x) = y[0] + y[1] x + ... + y[n-1] x^{n-1}$$

            Here we take $y[0], ..., y[n-1]$ to be the next $n$ outputs, which in this implementation are exactly the
            current state reversed.

            Let $P(x)$ be the characteristic polynomial of the LFSR. In the Galois model, the state polynomial
            $G(x)$ represents the element

            $$G(x) = g_0 + g_1 x + ... + g_{n-1} x^{n-1} \in GF(q)[x] / (P(x)),$$

            and one clock of the LFSR corresponds to multiplication by $x \mod P(x)$.

            $$G_{t+1}(x) = x G_t(x) \mod P(x)$$
            $$y[t] = \left\lfloor \frac{x G_t(x)}{P(x)} \right\rfloor$$

            If we start from an initial Galois state $G_0(x)$ and clock $n$ times, we have

            $$x^n G_0(x) = Y(x) P(x) + G_n(x),$$

            where $\deg(G_n) < n$. Taking the polynomial quotient by $x^n$ and using
            $\left\lfloor G_n(x) / x^n \right\rfloor = 0$, we obtain

            $$G_0(x) = \left\lfloor \frac{Y(x) P(x)}{x^n} \right\rfloor.$$

            This method constructs $Y(x)$ from the Fibonacci state, computes $G_0(x)$ from the formula above,
            and then uses the coefficients of $G_0(x)$ as the initial state of an equivalent Galois LFSR.

        Examples:
            Create a Fibonacci LFSR with a given initial state.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                fibonacci_lfsr = galois.FLFSR(feedback_poly, state=[1, 2, 3, 4])
                print(fibonacci_lfsr)

            Convert the Fibonacci LFSR to an equivalent Galois LFSR. Notice the initial state is different.

            .. ipython:: python

                galois_lfsr = fibonacci_lfsr.to_galois_lfsr()
                print(galois_lfsr)

            Step both LFSRs and see that their output sequences are identical.

            .. ipython:: python

                fibonacci_lfsr.step(10)
                galois_lfsr.step(10)
        """
        n = self.order

        # Y(x): output-block polynomial.
        # The Fibonacci state S = [S_0, ..., S_{n-1}] holds the next n outputs in *reverse* order: [y[n-1], ..., y[0]].
        # Reversing gives [y[0], ..., y[n-1]], which we interpret as
        #   Y(x) = y[0] + y[1] x + ... + y[n-1] x^{n-1}.
        Y = Poly(self.state[::-1])

        # P(x): characteristic polynomial of the LFSR.
        P = self.characteristic_poly

        # x: the monomial x in GF(q)[x].
        x = Poly.Identity(self.field)

        # G_0(x) = floor( Y(x) P(x) / x^n ).
        G0_poly = Y * P // (x**n)

        # Extract the first n coefficients of G_0(x) in ascending order:
        #   G_0(x) = g_0 + g_1 x + ... + g_{n-1} x^{n-1}.
        g0 = G0_poly.coefficients(n, order="asc")

        # Construct the equivalent Galois LFSR with feedback polynomial f(x)
        # (same feedback/connection polynomial) and initial state g0.
        return GLFSR(self.feedback_poly, state=g0)

    @property
    def field(self) -> Type[FieldArray]:
        """
        The :obj:`~galois.FieldArray` subclass for the finite field that defines the linear arithmetic.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly); lfsr
                lfsr.field
        """
        return super().field

    @property
    def feedback_poly(self) -> Poly:
        r"""
        The feedback polynomial $f(x) = 1 + a_1 x + a_2 x^2 + \dots + a_{n} x^{n}$.

        Notes:
            The feedback polynomial is the reciprocal of the characteristic polynomial $f(x) = x^n c(x^{-1})$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly); lfsr
                lfsr.feedback_poly
                assert lfsr.feedback_poly == lfsr.characteristic_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().feedback_poly

    @property
    def characteristic_poly(self) -> Poly:
        r"""
        The characteristic polynomial $c(x) = x^{n} + a_1 x^{n-1} + a_2 x^{n-2} + \dots + a_{n}$.

        Notes:
            The characteristic polynomial is the reciprocal of the feedback polynomial $c(x) = x^n f(x^{-1})$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly); lfsr
                lfsr.characteristic_poly
                assert lfsr.characteristic_poly == lfsr.feedback_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().characteristic_poly

    @property
    def taps(self) -> FieldArray:
        r"""
        The shift register taps $T = [-a_1, -a_2, \dots, -a_{n-1}, -a_n]$. The taps of the shift register define
        the linear recurrence relation.

        Examples:
            .. ipython:: python

                characteristic_poly = galois.primitive_poly(7, 4); characteristic_poly
                taps = -characteristic_poly.coeffs[1:]; taps
                lfsr = galois.FLFSR.Taps(taps)
                print(lfsr)
        """
        return super().taps

    @property
    def order(self) -> int:
        """
        The order of the linear recurrence and linear recurrent sequence. The order of a sequence is defined by the
        degree of the connection, feedback, and characteristic polynomials that generate it.
        """
        return super().order

    @property
    def initial_state(self) -> FieldArray:
        r"""
        The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.initial_state

            The initial state is unaffected as the Fibonacci LFSR is stepped.

            .. ipython:: python

                lfsr.step(10)
                lfsr.initial_state

        Group:
            State

        Order:
            62
        """
        return super().initial_state

    @property
    def state(self) -> FieldArray:
        r"""
        The current state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.FLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.state

            The current state is modified as the Fibonacci LFSR is stepped.

            .. ipython:: python

                lfsr.step(10)
                lfsr.state

        Group:
            State

        Order:
            62
        """
        return super().state


class fibonacci_lfsr_step_forward_jit(Function):
    """
    Steps the Fibonacci LFSR `steps` forward.

    .. code-block:: text
        :caption: Fibonacci LFSR Configuration

           y[t]
            +--------------+<-------------+<-------------+<-------------+
            |              ^              ^              ^              |
            |              | -a_1         | -a_2         | -a_{n-1}     | -a_n
            |              | T[0]         | T[1]         | T[n-2]       | T[n-1]
            |  +--------+  |  +--------+  |              |  +--------+  |
            +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t-n]
               +--------+     +--------+                    +--------+
                 y[t-1]         y[t-2]                        y[t-n]

    Arguments:
        taps: The set of taps T = [-a_1, -a_2, ..., -a_{n-1}, -a_n].
        state: The state vector [S_0, S_1, ..., S_{n-2}, S_{n-1}]. State will be modified in-place.
        steps: The number of output symbols to produce.

    Returns:
        The output sequence of size `steps`.
    """

    def __call__(self, taps, state, steps):
        if self.field.ufunc_mode != "python-calculate":
            state_ = state.astype(np.int64)  # NOTE: This will be modified
            y = self.jit(taps.astype(np.int64), state_, steps)
            y = y.astype(state.dtype)
        else:
            state_ = state.view(np.ndarray)  # NOTE: This will be modified
            y = self.python(taps.view(np.ndarray), state_, steps)
        y = self.field._view(y)

        return y, state_

    def set_globals(self):
        global ADD, MULTIPLY
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:], int64))

    @staticmethod
    def implementation(taps, state, steps):
        n = taps.size
        y = np.zeros(steps, dtype=state.dtype)  # The output array

        for i in range(steps):
            f = 0  # The feedback value
            for j in range(n):
                f = ADD(f, MULTIPLY(state[j], taps[j]))

            y[i] = state[-1]  # Output is popped off the shift register
            state[1:] = state[0:-1]  # Shift state rightward
            state[0] = f  # Insert feedback value at leftmost position

        return y


class fibonacci_lfsr_step_backward_jit(Function):
    """
    Steps the Fibonacci LFSR `steps` backward.

    .. code-block:: text
        :caption: Fibonacci LFSR Configuration

           y[t]
            +--------------+<-------------+<-------------+<-------------+
            |              ^              ^              ^              |
            |              | -a_1         | -a_2         | -a_{n-1}     | -a_n
            |              | T[0]         | T[1]         | T[n-2]       | T[n-1]
            |  +--------+  |  +--------+  |              |  +--------+  |
            +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t-n]
               +--------+     +--------+                    +--------+
                 y[t-1]         y[t-2]                        y[t-n]

    Arguments:
        taps: The set of taps T = [-a_1, -a_2, ..., -a_{n-1}, -a_n].
        state: The state vector [S_0, S_1, ..., S_{n-2}, S_{n-1}]. State will be modified in-place.
        steps: The number of output symbols to produce.

    Returns:
        The output sequence of size `steps`.
    """

    def __call__(self, taps, state, steps):
        if self.field.ufunc_mode != "python-calculate":
            state_ = state.astype(np.int64)  # NOTE: This will be modified
            y = self.jit(taps.astype(np.int64), state_, steps)
            y = y.astype(state.dtype)
        else:
            state_ = state.view(np.ndarray)  # NOTE: This will be modified
            y = self.python(taps.view(np.ndarray), state_, steps)
        y = self.field._view(y)

        return y, state_

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:], int64))

    @staticmethod
    def implementation(taps, state, steps):
        n = taps.size
        y = np.zeros(steps, dtype=state.dtype)  # The output array

        for i in range(steps):
            f = state[0]  # The feedback value
            state[0:-1] = state[1:]  # Shift state leftward

            s = f  # The unknown previous state value
            for j in range(n - 1):
                s = SUBTRACT(s, MULTIPLY(state[j], taps[j]))
            s = MULTIPLY(s, RECIPROCAL(taps[n - 1]))

            y[i] = s  # The previous output was the last value in the shift register
            state[-1] = s  # Assign recovered state to the leftmost position

        return y


###############################################################################
# Galois LFSR
###############################################################################


@export
class GLFSR(_LFSR):
    r"""
    A Galois linear-feedback shift register (LFSR).

    Notes:
        A Galois LFSR is defined by its feedback (connection) polynomial

        $$
        f(x) = 1 + a_1 x + a_2 x^2 + \dots + a_{n} x^{n},
        $$

        where $f(0) = 1$ and the degree $n$ equals the length of the shift register. The associated output sequence
        $y[t]$ satisfies the linear recurrence

        $$
        y[t] + a_1 y[t-1] + a_2 y[t-2] + \dots + a_{n} y[t-n] = 0.
        $$

        The characteristic polynomial of the sequence is the reciprocal of the feedback polynomial

        $$
        c(x) &= x^{n} + a_1 x^{n-1} + a_2 x^{n-2} + \dots + a_{n} \\
        &= x^{n} f(x^{-1}).
        $$

        In the Galois configuration, the shift register is arranged so that its taps implement the recurrence
        directly. The taps are simply the feedback coefficients $[-a_n, -a_{n-1}, \dots, -a_1]$ in a fixed left-to-right
        order that matches the chosen shift direction.

        .. code-block:: text
           :caption: Galois LFSR Configuration

            +--------------+<-------------+<-------------+<-------------+
            |              |              |              |              |
            | -a_n         | -a_{n-1}     | -a_{n-2}     | -a_1         |
            | T[0]         | T[1]         | T[2]         | T[n-1]       |
            |  +--------+  v  +--------+  v              v  +--------+  |
            +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
               +--------+     +--------+                    +--------+
                                                              y[t+1]

        The shift register taps $T$ are defined left-to-right as $T = [T_0, T_1, \dots, T_{n-2}, T_{n-1}]$.
        The state vector $S$ is also defined left-to-right as $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

    References:
        - Gardner, D. 2019. “Applications of the Galois Model LFSR in Cryptography”. figshare.
          https://hdl.handle.net/2134/21932.

    See Also:
        berlekamp_massey

    Examples:
        .. md-tab-set::

            .. md-tab-item:: GF(2)

                Create a Galois LFSR from a degree-4 primitive characteristic polynomial over $\mathrm{GF}(2)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(2, 4).reverse(); feedback_poly
                    lfsr = galois.GLFSR(feedback_poly)
                    print(lfsr)

                Step the Galois LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

            .. md-tab-item:: GF(p)

                Create a Galois LFSR from a degree-4 primitive characteristic polynomial over
                $\mathrm{GF}(7)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                    lfsr = galois.GLFSR(feedback_poly)
                    print(lfsr)

                Step the Galois LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

            .. md-tab-item:: GF(2^m)

                Create a Galois LFSR from a degree-4 primitive characteristic polynomial over
                $\mathrm{GF}(2^3)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(2**3, 4).reverse(); feedback_poly
                    lfsr = galois.GLFSR(feedback_poly)
                    print(lfsr)

                Step the Galois LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

            .. md-tab-item:: GF(p^m)

                Create a Galois LFSR from a degree-4 primitive characteristic polynomial over
                $\mathrm{GF}(3^3)$.

                .. ipython:: python

                    feedback_poly = galois.primitive_poly(3**3, 4).reverse(); feedback_poly
                    lfsr = galois.GLFSR(feedback_poly)
                    print(lfsr)

                Step the Galois LFSR and produce 10 output symbols.

                .. ipython:: python

                    lfsr.state
                    lfsr.step(10)
                    lfsr.state

    Group:
        linear-sequences
    """

    _type = "galois"

    def __init__(
        self,
        feedback_poly: Poly,
        state: ArrayLike | None = None,
    ):
        r"""
        Constructs a Galois LFSR from its feedback polynomial $f(x)$.

        Arguments:
            feedback_poly: The feedback polynomial $f(x) = 1 + a_1 x + a_2 x^2 + \dots + a_{n} x^{n}$.
            state: The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None`
                which corresponds to all ones.

        See Also:
            irreducible_poly, primitive_poly

        Notes:
            A Galois LFSR may be constructed from its characteristic polynomial $c(x)$ by passing in its
            reciprocal as the feedback polynomial. This is because $f(x) = x^n c(x^{-1})$.
        """
        super().__init__(feedback_poly, state=state)

    @classmethod
    def Taps(cls, taps: FieldArray, state: ArrayLike | None = None) -> Self:
        r"""
        Constructs a Galois LFSR from its taps $T = [-a_n, \dots, -a_2, -a_1]$.

        Arguments:
            taps: The shift register taps $T = [-a_n, \dots, -a_2, -a_1]$.
            state: The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None`
                which corresponds to all ones.

        Returns:
            A Galois LFSR with taps $T = [-a_n, \dots, -a_2, -a_1]$.

        Examples:
            .. ipython:: python

                characteristic_poly = galois.primitive_poly(7, 4); characteristic_poly
                taps = -characteristic_poly.coeffs[1:]; taps
                lfsr = galois.GLFSR.Taps(taps)
                print(lfsr)
        """
        return super().Taps(taps, state=state)

    def __repr__(self) -> str:
        """
        A terse representation of the Galois LFSR.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly)
                lfsr
        """
        with printoptions(coeffs="asc"):
            return f"<Galois LFSR: f(x) = {self.feedback_poly} over {self.field.name}>"

    def __str__(self) -> str:
        """
        A formatted string of relevant properties of the Galois LFSR.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly)
                print(lfsr)
        """
        string = "Galois LFSR:"
        string += f"\n  field: {self.field.name}"
        with printoptions(coeffs="asc"):
            string += f"\n  feedback_poly: {self.feedback_poly}"
        string += f"\n  characteristic_poly: {self.characteristic_poly}"
        string += f"\n  taps: {self.taps}"
        string += f"\n  order: {self.order}"
        string += f"\n  state: {self.state}"
        string += f"\n  initial_state: {self.initial_state}"

        return string

    def reset(self, state: ArrayLike | None = None):
        r"""
        Resets the Galois LFSR state to the specified state.

        Arguments:
            state: The state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None` which
                corresponds to the initial state.

        Examples:
            Step the Galois LFSR 10 steps to modify its state.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly); lfsr
                lfsr.state
                lfsr.step(10)
                lfsr.state

            Reset the Galois LFSR state.

            .. ipython:: python

                lfsr.reset()
                lfsr.state

            Create an Galois LFSR and view its initial state.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly); lfsr
                lfsr.state

            Reset the Galois LFSR state to a new state.

            .. ipython:: python

                lfsr.reset([1, 2, 3, 4])
                lfsr.state
        """
        return super().reset(state)

    def step(self, steps: int = 1) -> FieldArray:
        """
        Produces the next `steps` output symbols.

        Arguments:
            steps: The direction and number of output symbols to produce. The default is 1. If negative, the
                Galois LFSR will step backwards.

        Returns:
            An array of output symbols of type :obj:`field` with size `abs(steps)`.

        Examples:
            Step the Galois LFSR one output at a time.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                # Ending state
                lfsr.state

            Step the Galois LFSR 5 steps in one call. This is more efficient than iterating one output at a time.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.state
                lfsr.step(5)
                # Ending state
                lfsr.state

            Step the Galois LFSR 5 steps backward. Notice the output sequence is the reverse of the original sequence.
            Also notice the ending state is the same as the initial state.

            .. ipython:: python

                lfsr.step(-5)
                lfsr.state
        """
        return super().step(steps)

    def to_fibonacci_lfsr(self) -> FLFSR:
        r"""
        Converts the Galois LFSR to a Fibonacci LFSR that produces the same output.

        Returns:
            An equivalent Fibonacci LFSR.

        Notes:
            To construct an equivalent Fibonacci LFSR, we use the fact that a Fibonacci LFSR with
            feedback polynomial $f(x)$ and initial state

            $$S = [y[n-1], \dots, y[1], y[0]]$$

            will produce the sequence $y[0], y[1], \dots, y[n-1], \dots$.

            This method therefore:

            1. Takes the next $n$ outputs $y[0], \dots, y[n-1]$ of the Galois LFSR.
            2. Forms the Fibonacci initial state $S = [y[n-1], \dots, y[0]]$.
            3. Constructs a Fibonacci LFSR with the same feedback polynomial $f(x)$ and state $S$.

            The Galois LFSR is stepped forward $n$ times to obtain these outputs and then stepped
            backward $n$ times, so its state is unchanged.

        Examples:
            Create a Galois LFSR with a given initial state.

            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                galois_lfsr = galois.GLFSR(feedback_poly, state=[1, 2, 3, 4])
                print(galois_lfsr)

            Convert the Galois LFSR to an equivalent Fibonacci LFSR. Notice the initial state is different.

            .. ipython:: python

                fibonacci_lfsr = galois_lfsr.to_fibonacci_lfsr()
                print(fibonacci_lfsr)

            Step both LFSRs and see that their output sequences are identical.

            .. ipython:: python

                galois_lfsr.step(10)
                fibonacci_lfsr.step(10)
        """
        output = self.step(self.order)
        state = output[::-1]
        self.step(-self.order)

        # Create a new object so the initial state is set properly
        return FLFSR(self.feedback_poly, state=state)

    @property
    def field(self) -> Type[FieldArray]:
        """
        The :obj:`~galois.FieldArray` subclass for the finite field that defines the linear arithmetic.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly); lfsr
                lfsr.field
        """
        return super().field

    @property
    def feedback_poly(self) -> Poly:
        r"""
        The feedback polynomial $f(x) = 1 + a_1 x + a_2 x^2 + \dots + a_{n} x^{n}$.

        Notes:
            The feedback polynomial is the reciprocal of the characteristic polynomial $f(x) = x^n c(x^{-1})$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly); lfsr
                lfsr.feedback_poly
                assert lfsr.feedback_poly == lfsr.characteristic_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().feedback_poly

    @property
    def characteristic_poly(self) -> Poly:
        r"""
        The characteristic polynomial $c(x) = x^{n} + a_1 x^{n-1} + a_2 x^{n-2} + \dots + a_{n}$.

        Notes:
            The characteristic polynomial is the reciprocal of the feedback polynomial $c(x) = x^n f(x^{-1})$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly); lfsr
                lfsr.characteristic_poly
                assert lfsr.characteristic_poly == lfsr.feedback_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().characteristic_poly

    @property
    def taps(self) -> FieldArray:
        r"""
        The shift register taps $T = [-a_n, \dots, -a_2, -a_1]$. The taps of the shift register define
        the linear recurrence relation.

        Examples:
            .. ipython:: python

                characteristic_poly = galois.primitive_poly(7, 4); characteristic_poly
                taps = -characteristic_poly.coeffs[1:]; taps
                lfsr = galois.GLFSR.Taps(taps)
                print(lfsr)
        """
        return super().taps

    @property
    def order(self) -> int:
        """
        The order of the linear recurrence and linear recurrent sequence. The order of a sequence is defined by the
        degree of the connection, feedback, and characteristic polynomials that generate it.
        """
        return super().order

    @property
    def initial_state(self) -> FieldArray:
        r"""
        The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.initial_state

            The initial state is unaffected as the Galois LFSR is stepped.

            .. ipython:: python

                lfsr.step(10)
                lfsr.initial_state

        Group:
            State

        Order:
            62
        """
        return super().initial_state

    @property
    def state(self) -> FieldArray:
        r"""
        The current state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        Examples:
            .. ipython:: python

                feedback_poly = galois.primitive_poly(7, 4).reverse(); feedback_poly
                lfsr = galois.GLFSR(feedback_poly, state=[1, 2, 3, 4]); lfsr
                lfsr.state

            The current state is modified as the Galois LFSR is stepped.

            .. ipython:: python

                lfsr.step(10)
                lfsr.state

        Group:
            State

        Order:
            62
        """
        return super().state


class galois_lfsr_step_forward_jit(Function):
    """
    Steps the Galois LFSR `steps` forward.

    .. code-block:: text
        :caption: Galois LFSR Configuration

        +--------------+<-------------+<-------------+<-------------+
        |              |              |              |              |
        | -a_n         | -a_{n-1}     | -a_{n-2}     | -a_1         |
        | T[0]         | T[1]         | T[2]         | T[n-1]       |
        |  +--------+  v  +--------+  v              v  +--------+  |
        +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
           +--------+     +--------+                    +--------+
                                                          y[t+1]

    Arguments:
        taps: The set of taps T = [-a_n, -a_{n-1}, ..., -a_2, -a_1].
        state: The state vector [S_0, S_1, ..., S_{n-2}, S_{n-1}]. State will be modified in-place.
        steps: The number of output symbols to produce.

    Returns:
        The output sequence of size `steps`.
    """

    def __call__(self, taps, state, steps):
        if self.field.ufunc_mode != "python-calculate":
            state_ = state.astype(np.int64)  # NOTE: This will be modified
            y = self.jit(taps.astype(np.int64), state_, steps)
            y = y.astype(state.dtype)
        else:
            state_ = state.view(np.ndarray)  # NOTE: This will be modified
            y = self.python(taps.view(np.ndarray), state_, steps)
        y = self.field._view(y)

        return y, state_

    def set_globals(self):
        global ADD, MULTIPLY
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:], int64))

    @staticmethod
    def implementation(taps, state, steps):
        n = taps.size
        y = np.zeros(steps, dtype=state.dtype)  # The output array

        for i in range(steps):
            f = state[n - 1]  # The feedback value
            y[i] = f  # The output

            if f == 0:
                state[1:] = state[0:-1]
                state[0] = 0
            else:
                for j in range(n - 1, 0, -1):
                    state[j] = ADD(state[j - 1], MULTIPLY(f, taps[j]))
                state[0] = MULTIPLY(f, taps[0])

        return y


class galois_lfsr_step_backward_jit(Function):
    """
    Steps the Galois LFSR `steps` backward.

    .. code-block:: text
        :caption: Galois LFSR Configuration

        +--------------+<-------------+<-------------+<-------------+
        |              |              |              |              |
        | -a_n         | -a_{n-1}     | -a_{n-2}     | -a_1         |
        | T[0]         | T[1]         | T[2]         | T[n-1]       |
        |  +--------+  v  +--------+  v              v  +--------+  |
        +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
           +--------+     +--------+                    +--------+
                                                          y[t+1]

    Arguments:
        taps: The set of taps T = [-a_n, -a_{n-1}, ..., -a_2, -a_1].
        state: The state vector [S_0, S_1, ..., S_{n-2}, S_{n-1}]. State will be modified in-place.
        steps: The number of output symbols to produce.

    Returns:
        The output sequence of size `steps`.
    """

    def __call__(self, taps, state, steps):
        if self.field.ufunc_mode != "python-calculate":
            state_ = state.astype(np.int64)  # NOTE: This will be modified
            y = self.jit(taps.astype(np.int64), state_, steps)
            y = y.astype(state.dtype)
        else:
            state_ = state.view(np.ndarray)  # NOTE: This will be modified
            y = self.python(taps.view(np.ndarray), state_, steps)
        y = self.field._view(y)

        return y, state_

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:], int64))

    @staticmethod
    def implementation(taps, state, steps):
        n = taps.size
        y = np.zeros(steps, dtype=state.dtype)  # The output array

        for i in range(steps):
            f = MULTIPLY(state[0], RECIPROCAL(taps[0]))  # The feedback value

            for j in range(0, n - 1):
                state[j] = SUBTRACT(state[j + 1], MULTIPLY(f, taps[j + 1]))

            state[n - 1] = f
            y[i] = f  # The output

        return y


###############################################################################
# Berlekamp-Massey algorithm
###############################################################################


@overload
def berlekamp_massey(sequence: FieldArray, output: Literal["characteristic"] = "characteristic") -> Poly: ...


@overload
def berlekamp_massey(sequence: FieldArray, output: Literal["connection"]) -> Poly: ...


@overload
def berlekamp_massey(sequence: FieldArray, output: Literal["fibonacci"]) -> FLFSR: ...


@overload
def berlekamp_massey(sequence: FieldArray, output: Literal["galois"]) -> GLFSR: ...


@export
def berlekamp_massey(sequence, output="characteristic"):
    r"""
    Finds the characteristic polynomial $c(x)$ of the input linear recurrent sequence using the
    Berlekamp-Massey algorithm.

    Arguments:
        sequence: A linear recurrent sequence $y$ in $\mathrm{GF}(p^m)$.
        output: The output object type.

            - `"characteristic"` (default): Returns the characteristic polynomial $c(x)$ that generates the linear
              recurrent sequence. This is equivalent to the minimal polynomial. The characteristic polynomial is the
              reciprocal of the connection polynomial, $c(x) = x^{n} C(x^{-1})$.
            - `"connection"`: Returns the connection polynomial $C(x)$ that generates the linear recurrent
              sequence. The connection polynomial is equivalent to the feedback polynomial $f(x)$ of an LFSR.
            - `"fibonacci"`: Returns a Fibonacci LFSR whose next $n$ outputs produce $y$.
            - `"galois"`: Returns a Galois LFSR whose next $n$ outputs produce $y$.

    Returns:
        The characteristic (minimal) polynomial $c(x)$, the connection polynomial $C(x)$, a Fibonacci LFSR,
        or a Galois LFSR, depending on the value of `output`.

    Notes:
        The characteristic polynomial of a linear recurrent sequence is defined as

        $$
        c(x) &= x^{n} + a_1 x^{n-1} + a_2 x^{n-2} + \dots + a_{n} \\
        &= x^{n} f(x^{-1}).
        $$

        The connection polynomial $C(x)$ is defined as

        $$
        C(x) &= 1 + a_1 x + a_2 x^2 + \dots + a_{n} x^{n} \\
        &= f(x) = x^{n} c(x^{-1}),
        $$

        where $C(0) = f(0) = 1$ and the degree $n$ equals the length of the shift register. The associated output
        sequence $y[t]$ satisfies the linear recurrence

        $$
        y[t] + a_1 y[t-1] + a_2 y[t-2] + \dots + a_{n} y[t-n] = 0.
        $$

        For a linear recurrent sequence with order $n$, at least $2n$ output symbols are required to determine the
        minimal polynomial.

    References:
        - Gardner, D. 2019. “Applications of the Galois Model LFSR in Cryptography”. https://hdl.handle.net/2134/21932.
        - Sachs, J. Linear Feedback Shift Registers for the Uninitiated, Part VI: Sing Along with the
          Berlekamp-Massey Algorithm. https://www.embeddedrelated.com/showarticle/1099.php
        - https://crypto.stanford.edu/~mironov/cs359/massey.pdf

    Examples:
        The sequence below is a degree-4 linear recurrent sequence over $\mathrm{GF}(7)$.

        .. ipython:: python

            GF = galois.GF(7)
            y = GF([5, 5, 1, 3, 1, 4, 6, 6, 5, 5])

        The characteristic polynomial is $c(x) = x^4 + x^2 + 3x + 5$ over $\mathrm{GF}(7)$.

        .. ipython:: python

            galois.berlekamp_massey(y)

        The connection (feedback) polynomial is $C(x) = 5x^4 + 3x^3 + x^2 + 1$ over $\mathrm{GF}(7)$.

        .. ipython:: python

            galois.berlekamp_massey(y, output="connection")

        Use the Berlekamp-Massey algorithm to return equivalent Fibonacci LFSR that reproduces the sequence.

        .. ipython:: python

            lfsr = galois.berlekamp_massey(y, output="fibonacci")
            print(lfsr)
            z = lfsr.step(y.size); z
            assert np.array_equal(y, z)

        Use the Berlekamp-Massey algorithm to return equivalent Galois LFSR that reproduces the sequence.

        .. ipython:: python

            lfsr = galois.berlekamp_massey(y, output="galois")
            print(lfsr)
            z = lfsr.step(y.size); z
            assert np.array_equal(y, z)

    Group:
        linear-sequences
    """
    verify_isinstance(sequence, FieldArray)
    verify_isinstance(output, str)
    if not sequence.ndim == 1:
        raise ValueError(f"Argument 'sequence' must be 1-D, not {sequence.ndim}-D.")
    if not output in ["characteristic", "connection", "fibonacci", "galois"]:
        raise ValueError(
            f"Argument 'output' must be in ['characteristic', 'connection', 'fibonacci', 'galois'], not {output!r}."
        )

    field = type(sequence)
    coeffs = berlekamp_massey_jit(field)(sequence)  # Connection polynomial coefficients, degree-descending
    connection_poly = Poly(coeffs, field=field)

    if output == "characteristic":
        return connection_poly.reverse()
    if output == "connection":
        return connection_poly

    # The first n outputs are the Fibonacci state reversed
    state = sequence[0 : connection_poly.degree][::-1]
    fibonacci_lfsr = FLFSR(connection_poly, state=state)
    if output == "fibonacci":
        return fibonacci_lfsr
    else:
        return fibonacci_lfsr.to_galois_lfsr()


class berlekamp_massey_jit(Function):
    """
    Finds the connection polynomial C(x) (in degree-descending order) of the input linear recurrent sequence
    using the Berlekamp-Massey algorithm.
    """

    def __call__(self, sequence):
        if self.field.ufunc_mode != "python-calculate":
            coeffs = self.jit(sequence.astype(np.int64))
            coeffs = coeffs.astype(sequence.dtype)
        else:
            coeffs = self.python(sequence.view(np.ndarray))
        coeffs = self.field._view(coeffs)

        return coeffs

    def set_globals(self):
        global ADD, SUBTRACT, MULTIPLY, RECIPROCAL
        ADD = self.field._add.ufunc_call_only
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:]))

    @staticmethod
    def implementation(sequence):  # pragma: no cover
        S = sequence  # The input sequence, S = S0 + S1*x + S2*x^2 + ...
        C = np.zeros_like(S)
        C[0] = 1  # The current connection polynomial C(x) = 1
        B = np.zeros_like(S)
        B[0] = 1  # The best connection polynomial B(x) = 1
        L = 0  # The current linear complexity
        m = 1  # The number of steps since last update
        b = 1  # The last discrepancy

        for n in range(0, S.size):
            d = 0  # The discrepancy at step n, d = Sn*1 + C1*Sn-1 + ... + CL*Sn-L
            for i in range(0, L + 1):
                d = ADD(d, MULTIPLY(S[n - i], C[i]))

            if d == 0:
                # The current C(x) is still valid, no update needed
                m += 1
            else:
                # The current recurrence fails, need to update C(x)
                if 2 * L > n:
                    # There is room to adjust C(x) without increasing its degree
                    # Update C(x) := C(x) - d/b * x^m * B(x)
                    d_over_b = MULTIPLY(d, RECIPROCAL(b))
                    for i in range(m, S.size):
                        C[i] = SUBTRACT(C[i], MULTIPLY(d_over_b, B[i - m]))

                    # fixed_d = 0
                    # for i in range(0, L + 1):
                    #     fixed_d = ADD(fixed_d, MULTIPLY(S[n - i], C[i]))
                    # assert fixed_d == 0, "Berlekamp-Massey algorithm failure: discrepancy should be zero after update."

                    m += 1
                else:
                    # The current recurrence is too short. A longer recurrence is needed. Update L and B(x) = C(x).
                    T = C.copy()
                    d_over_b = MULTIPLY(d, RECIPROCAL(b))
                    for i in range(m, S.size):
                        C[i] = SUBTRACT(C[i], MULTIPLY(d_over_b, B[i - m]))
                    L = n + 1 - L  # New new linear complexity
                    B = T.copy()
                    b = d  # Last discrepancy
                    m = 1  # Reset steps since last update

        # C is the connection polynomial C(x) = 1 + C1*x + C2*x^2 + ... + CL*x^L
        C = C[: L + 1]

        # Trim trailing, high-degree zeros
        idxs = np.where(C != 0)[0]
        if idxs.size > 0:
            C = C[: idxs[-1] + 1]
        else:
            C = C[:1]  # C(x) = 0 polynomial

        return C[::-1]  # Return C(x) coefficients in degree-descending order
