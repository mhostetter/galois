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
            # T = [c_n-1, c_n-2, ..., c_1, c_0]
            # c(x) = x^{n} - c_{n-1}x^{n-1} - c_{n-2}x^{n-2} - \dots - c_{1}x - c_{0}
            self._taps = -self.characteristic_poly.coeffs[1:]
        else:
            # T = [c_0, c_1, ..., c_n-2, c_n-1]
            # c(x) = x^{n} - c_{n-1}x^{n-1} - c_{n-2}x^{n-2} - \dots - c_{1}x - c_{0}
            self._taps = -self.characteristic_poly.coeffs[1:][::-1]

        if state is None:
            state = self.field.Ones(self.order)

        self._initial_state = self._verify_and_convert_state(state)
        self._state = self.initial_state.copy()

    @classmethod
    def Taps(cls, taps: FieldArray, state: ArrayLike | None = None) -> Self:
        verify_isinstance(taps, FieldArray)

        if cls._type == "fibonacci":
            # T = [c_n-1, c_n-2, ..., c_1, c_0]
            # f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1
            coeffs = -taps[::-1]
            coeffs = np.append(coeffs, 1)  # Add x^0 term
            feedback_poly = Poly(coeffs)
        else:
            # T = [c_0, c_1, ..., c_n-2, c_n-1]
            # f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1
            coeffs = -taps
            coeffs = np.append(coeffs, 1)  # Add x^0 term
            feedback_poly = Poly(coeffs)

        return cls(feedback_poly, state=state)

    def _verify_and_convert_state(self, state: ArrayLike):
        verify_isinstance(state, (tuple, list, np.ndarray, FieldArray))

        state = self.field(state)  # Coerce array-like object to field array

        # if not state.size == self.order:
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

        if not self.characteristic_poly.coeffs[-1] > 0:
            raise ValueError(
                "Can only step the shift register backwards if the c_0 tap is non-zero, "
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
        A Fibonacci LFSR is defined by its feedback polynomial $f(x)$.

        $$f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1 = x^n c(x^{-1})$$

        The feedback polynomial is the reciprocal of the characteristic polynomial $c(x)$ of the linear recurrent
        sequence $y$ produced by the Fibonacci LFSR.

        $$c(x) = x^{n} - c_{n-1}x^{n-1} - c_{n-2}x^{n-2} - \dots - c_{1}x - c_{0}$$

        $$y_t = c_{n-1}y_{t-1} + c_{n-2}y_{t-2} + \dots + c_{1}y_{t-n+2} + c_{0}y_{t-n+1}$$

        .. code-block:: text
           :caption: Fibonacci LFSR Configuration

            +--------------+<-------------+<-------------+<-------------+
            |              ^              ^              ^              |
            |              | c_n-1        | c_n-2        | c_1          | c_0
            |              | T[0]         | T[1]         | T[n-2]       | T[n-1]
            |  +--------+  |  +--------+  |              |  +--------+  |
            +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
               +--------+     +--------+                    +--------+
                y[t+n-1]       y[t+n-2]                       y[t+1]

        The shift register taps $T$ are defined left-to-right as $T = [T_0, T_1, \dots, T_{n-2}, T_{n-1}]$.
        The state vector $S$ is also defined left-to-right as $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        In the Fibonacci configuration, the shift register taps are $T = [c_{n-1}, c_{n-2}, \dots, c_1, c_0]$.
        Additionally, the state vector is equal to the next $n$ outputs in reversed order, namely
        $S = [y_{t+n-1}, y_{t+n-2}, \dots, y_{t+2}, y_{t+1}]$.

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

                    c = galois.primitive_poly(2, 4); c
                    lfsr = galois.FLFSR(c.reverse())
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

                    c = galois.primitive_poly(7, 4); c
                    lfsr = galois.FLFSR(c.reverse())
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

                    c = galois.primitive_poly(2**3, 4); c
                    lfsr = galois.FLFSR(c.reverse())
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

                    c = galois.primitive_poly(3**3, 4); c
                    lfsr = galois.FLFSR(c.reverse())
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
            feedback_poly: The feedback polynomial
                $f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1$.
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
        Constructs a Fibonacci LFSR from its taps $T = [c_{n-1}, c_{n-2}, \dots, c_1, c_0]$.

        Arguments:
            taps: The shift register taps $T = [c_{n-1}, c_{n-2}, \dots, c_1, c_0]$.
            state: The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None`
                which corresponds to all ones.

        Returns:
            A Fibonacci LFSR with taps $T = [c_{n-1}, c_{n-2}, \dots, c_1, c_0]$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                taps = -c.coeffs[1:]; taps
                lfsr = galois.FLFSR.Taps(taps)
                print(lfsr)
        """
        return super().Taps(taps, state=state)

    def __repr__(self) -> str:
        """
        A terse representation of the Fibonacci LFSR.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.FLFSR(c.reverse())
                lfsr
        """
        return f"<Fibonacci LFSR: f(x) = {self.feedback_poly} over {self.field.name}>"

    def __str__(self) -> str:
        """
        A formatted string of relevant properties of the Fibonacci LFSR.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.FLFSR(c.reverse())
                print(lfsr)
        """
        string = "Fibonacci LFSR:"
        string += f"\n  field: {self.field.name}"
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

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.FLFSR(c.reverse()); lfsr
                lfsr.state
                lfsr.step(10)
                lfsr.state

            Reset the Fibonacci LFSR state.

            .. ipython:: python

                lfsr.reset()
                lfsr.state

            Create an Fibonacci LFSR and view its initial state.

            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.FLFSR(c.reverse()); lfsr
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

                c = galois.primitive_poly(7, 4)
                lfsr = galois.FLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                # Ending state
                lfsr.state

            Step the Fibonacci LFSR 5 steps in one call. This is more efficient than iterating one output at a time.

            .. ipython:: python

                c = galois.primitive_poly(7, 4)
                lfsr = galois.FLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
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
        """
        Converts the Fibonacci LFSR to a Galois LFSR that produces the same output.

        Returns:
            An equivalent Galois LFSR.

        Examples:
            Create a Fibonacci LFSR with a given initial state.

            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                fibonacci_lfsr = galois.FLFSR(c.reverse(), state=[1, 2, 3, 4])
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
        # See answer to this question https://crypto.stackexchange.com/questions/60634/lfsr-jump-ahead-algorithm

        # The Fibonacci LFSR state F = [f_0, f_1, ..., f_n-1] has the next n outputs of y = [f_n-1, ..., f_1, f_0]
        # This corresponds to the polynomial F(x) = f_n-1*x^n-1 + f_n-2*x^n-2 + ... + f_1*x + f_0
        F = Poly(self.state[::-1])  # Fibonacci output polynomial over GF(q)

        # The Galois LFSR state G = [g_0, g_1, ..., g_n-1] represents the element g_0 + g_1*x + ... + g_n-1*x^n-1
        # in GF(q^n). Let G_i and G_j indicate the state vector at times i and j. The next state G_j = G_i*x % P(x)
        # and y_j = G_i*x // P(x) for j = i + 1. This can be rearranged as G_i*x = y_j*P(x) + G_j. For the
        # Fibonacci LFSR output polynomial F(x), initial Galois LFSR state G_0, finial Galois LFSR state G_n,
        # and characteristic polynomial P(x), the equivalence may be written as G_0*x^n = F(x)*P(x) + G_n or
        # equivalently G_0 = (F(x)*P(x) + G_n) // x^n. The last equation simplifies to G_0 = F(x)*P(x) // x^n because
        # G_n has degree less than n, therefore G_n // x^n = 0.
        P = self.characteristic_poly
        S = F * P // Poly.Identity(self.field) ** self.order
        # state = S.coefficients(self.order)[::-1]  # Get coefficients in ascending order
        state = S.coefficients(self.order, order="asc")  # Get coefficients in ascending order

        return GLFSR(self.feedback_poly, state=state)

    @property
    def field(self) -> Type[FieldArray]:
        """
        The :obj:`~galois.FieldArray` subclass for the finite field that defines the linear arithmetic.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.FLFSR(c.reverse()); lfsr
                lfsr.field
        """
        return super().field

    @property
    def feedback_poly(self) -> Poly:
        r"""
        The feedback polynomial $f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1$
        that defines the feedback arithmetic.

        Notes:
            The feedback polynomial is the reciprocal of the characteristic polynomial $f(x) = x^n c(x^{-1})$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.FLFSR(c.reverse()); lfsr
                lfsr.feedback_poly
                lfsr.feedback_poly == lfsr.characteristic_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().feedback_poly

    @property
    def characteristic_poly(self) -> Poly:
        r"""
        The characteristic polynomial $c(x) = x^{n} - c_{n-1}x^{n-1} - c_{n-2}x^{n-2} - \dots - c_{1}x - c_{0}$
        that defines the linear recurrent sequence.

        Notes:
            The characteristic polynomial is the reciprocal of the feedback polynomial $c(x) = x^n f(x^{-1})$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.FLFSR(c.reverse()); lfsr
                lfsr.characteristic_poly
                lfsr.characteristic_poly == lfsr.feedback_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().characteristic_poly

    @property
    def taps(self) -> FieldArray:
        r"""
        The shift register taps $T = [c_{n-1}, c_{n-2}, \dots, c_1, c_0]$. The taps of the shift register define
        the linear recurrence relation.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                taps = -c.coeffs[1:]; taps
                lfsr = galois.FLFSR.Taps(taps); lfsr
                lfsr.taps
        """
        return super().taps

    @property
    def order(self) -> int:
        """
        The order of the linear recurrence/linear recurrent sequence. The order of a sequence is defined by the
        degree of the minimal polynomial that produces it.
        """
        return super().order

    @property
    def initial_state(self) -> FieldArray:
        r"""
        The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4)
                lfsr = galois.FLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
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

                c = galois.primitive_poly(7, 4)
                lfsr = galois.FLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
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

        +--------------+<-------------+<-------------+<-------------+
        |              ^              ^              ^              |
        |              | c_n-1        | c_n-2        | c_1          | c_0
        |              | T[0]         | T[1]         | T[n-2]       | T[n-1]
        |  +--------+  |  +--------+  |              |  +--------+  |
        +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
           +--------+     +--------+                    +--------+
            y[t+n-1]       y[t+n-2]                       y[t+1]

    Arguments:
        taps: The set of taps T = [c_n-1, c_n-2, ..., c_1, c_0].
        state: The state vector [S_0, S_1, ..., S_n-2, S_n-1]. State will be modified in-place!
        steps: The number of output symbols to produce.
        feedback: `True` indicates to output the feedback value `y_1[t]` (LRS) and `False` indicates to output the
            value out of the shift register `y_2[t]`.

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

        +--------------+<-------------+<-------------+<-------------+
        |              ^              ^              ^              |
        |              | c_n-1        | c_n-2        | c_1          | c_0
        |              | T[0]         | T[1]         | T[n-2]       | T[n-1]
        |  +--------+  |  +--------+  |              |  +--------+  |
        +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
           +--------+     +--------+                    +--------+
            y[t+n-1]       y[t+n-2]                       y[t+1]

    Arguments:
        taps: The set of taps T = [c_n-1, c_n-2, ..., c_1, c_0].
        state: The state vector [S_0, S_1, ..., S_n-2, S_n-1]. State will be modified in-place!
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
        A Galois LFSR is defined by its feedback polynomial $f(x)$.

        $$f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1 = x^n c(x^{-1})$$

        The feedback polynomial is the reciprocal of the characteristic polynomial $c(x)$ of the linear recurrent
        sequence $y$ produced by the Galois LFSR.

        $$c(x) = x^{n} - c_{n-1}x^{n-1} - c_{n-2}x^{n-2} - \dots - c_{1}x - c_{0}$$

        $$y_t = c_{n-1}y_{t-1} + c_{n-2}y_{t-2} + \dots + c_{1}y_{t-n+2} + c_{0}y_{t-n+1}$$

        .. code-block:: text
           :caption: Galois LFSR Configuration

            +--------------+<-------------+<-------------+<-------------+
            |              |              |              |              |
            | c_0          | c_1          | c_2          | c_n-1        |
            | T[0]         | T[1]         | T[2]         | T[n-1]       |
            |  +--------+  v  +--------+  v              v  +--------+  |
            +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
               +--------+     +--------+                    +--------+
                                                              y[t+1]

        The shift register taps $T$ are defined left-to-right as $T = [T_0, T_1, \dots, T_{n-2}, T_{n-1}]$.
        The state vector $S$ is also defined left-to-right as $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        In the Galois configuration, the shift register taps are $T = [c_0, c_1, \dots, c_{n-2}, c_{n-1}]$.

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

                    c = galois.primitive_poly(2, 4); c
                    lfsr = galois.GLFSR(c.reverse())
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

                    c = galois.primitive_poly(7, 4); c
                    lfsr = galois.GLFSR(c.reverse())
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

                    c = galois.primitive_poly(2**3, 4); c
                    lfsr = galois.GLFSR(c.reverse())
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

                    c = galois.primitive_poly(3**3, 4); c
                    lfsr = galois.GLFSR(c.reverse())
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
            feedback_poly: The feedback polynomial
                $f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1$.
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
        Constructs a Galois LFSR from its taps $T = [c_0, c_1, \dots, c_{n-2}, c_{n-1}]$.

        Arguments:
            taps: The shift register taps $T = [c_0, c_1, \dots, c_{n-2}, c_{n-1}]$.
            state: The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None`
                which corresponds to all ones.

        Returns:
            A Galois LFSR with taps $T = [c_0, c_1, \dots, c_{n-2}, c_{n-1}]$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                taps = -c.coeffs[1:][::-1]; taps
                lfsr = galois.GLFSR.Taps(taps)
                print(lfsr)
        """
        return super().Taps(taps, state=state)

    def __repr__(self) -> str:
        """
        A terse representation of the Galois LFSR.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.GLFSR(c.reverse())
                lfsr
        """
        return f"<Galois LFSR: f(x) = {self.feedback_poly} over {self.field.name}>"

    def __str__(self) -> str:
        """
        A formatted string of relevant properties of the Galois LFSR.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.GLFSR(c.reverse())
                print(lfsr)
        """
        string = "Galois LFSR:"
        string += f"\n  field: {self.field.name}"
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

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.GLFSR(c.reverse()); lfsr
                lfsr.state
                lfsr.step(10)
                lfsr.state

            Reset the Galois LFSR state.

            .. ipython:: python

                lfsr.reset()
                lfsr.state

            Create an Galois LFSR and view its initial state.

            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.GLFSR(c.reverse()); lfsr
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

                c = galois.primitive_poly(7, 4)
                lfsr = galois.GLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                lfsr.state, lfsr.step()
                # Ending state
                lfsr.state

            Step the Galois LFSR 5 steps in one call. This is more efficient than iterating one output at a time.

            .. ipython:: python

                c = galois.primitive_poly(7, 4)
                lfsr = galois.GLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
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
        """
        Converts the Galois LFSR to a Fibonacci LFSR that produces the same output.

        Returns:
            An equivalent Fibonacci LFSR.

        Examples:
            Create a Galois LFSR with a given initial state.

            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                galois_lfsr = galois.GLFSR(c.reverse(), state=[1, 2, 3, 4])
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

        # Create a new object so the intital state is set properly
        return FLFSR(self.feedback_poly, state=state)

    @property
    def field(self) -> Type[FieldArray]:
        """
        The :obj:`~galois.FieldArray` subclass for the finite field that defines the linear arithmetic.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.GLFSR(c.reverse()); lfsr
                lfsr.field
        """
        return super().field

    @property
    def feedback_poly(self) -> Poly:
        r"""
        The feedback polynomial $f(x) = -c_{0}x^{n} - c_{1}x^{n-1} - \dots - c_{n-2}x^{2} - c_{n-1}x + 1$
        that defines the feedback arithmetic.

        Notes:
            The feedback polynomial is the reciprocal of the characteristic polynomial $f(x) = x^n c(x^{-1})$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.GLFSR(c.reverse()); lfsr
                lfsr.feedback_poly
                lfsr.feedback_poly == lfsr.characteristic_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().feedback_poly

    @property
    def characteristic_poly(self) -> Poly:
        r"""
        The characteristic polynomial $c(x) = x^{n} - c_{n-1}x^{n-1} - c_{n-2}x^{n-2} - \dots - c_{1}x - c_{0}$
        that defines the linear recurrent sequence.

        Notes:
            The characteristic polynomial is the reciprocal of the feedback polynomial $c(x) = x^n f(x^{-1})$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                lfsr = galois.GLFSR(c.reverse()); lfsr
                lfsr.characteristic_poly
                lfsr.characteristic_poly == lfsr.feedback_poly.reverse()

        Group:
            Polynomials

        Order:
            61
        """
        return super().characteristic_poly

    @property
    def taps(self) -> FieldArray:
        r"""
        The shift register taps $T = [c_0, c_1, \dots, c_{n-2}, c_{n-1}]$. The taps of the shift register define
        the linear recurrence relation.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4); c
                taps = -c.coeffs[1:][::-1]; taps
                lfsr = galois.GLFSR.Taps(taps); lfsr
                lfsr.taps
        """
        return super().taps

    @property
    def order(self) -> int:
        """
        The order of the linear recurrence/linear recurrent sequence. The order of a sequence is defined by the
        degree of the minimal polynomial that produces it.
        """
        return super().order

    @property
    def initial_state(self) -> FieldArray:
        r"""
        The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$.

        Examples:
            .. ipython:: python

                c = galois.primitive_poly(7, 4)
                lfsr = galois.GLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
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

                c = galois.primitive_poly(7, 4)
                lfsr = galois.GLFSR(c.reverse(), state=[1, 2, 3, 4]); lfsr
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
        | c_0          | c_1          | c_2          | c_n-1        |
        | T[0]         | T[1]         | T[2]         | T[n-1]       |
        |  +--------+  v  +--------+  v              v  +--------+  |
        +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
           +--------+     +--------+                    +--------+
                                                          y[t+1]

    Arguments:
        taps: The set of taps T = [c_0, c_1, ..., c_n-2, c_n-2].
        state: The state vector [S_0, S_1, ..., S_n-2, S_n-1]. State will be modified in-place!
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
        | c_0          | c_1          | c_2          | c_n-1        |
        | T[0]         | T[1]         | T[2]         | T[n-1]       |
        |  +--------+  v  +--------+  v              v  +--------+  |
        +->|  S[0]  |--+->|  S[1]  |--+---  ...   ---+->| S[n-1] |--+--> y[t]
           +--------+     +--------+                    +--------+
                                                          y[t+1]

    Arguments:
        taps: The set of taps T = [c_0, c_1, ..., c_n-2, c_n-2].
        state: The state vector [S_0, S_1, ..., S_n-2, S_n-1]. State will be modified in-place!
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
def berlekamp_massey(sequence: FieldArray, output: Literal["minimal"] = "minimal") -> Poly: ...


@overload
def berlekamp_massey(sequence: FieldArray, output: Literal["fibonacci"]) -> FLFSR: ...


@overload
def berlekamp_massey(sequence: FieldArray, output: Literal["galois"]) -> GLFSR: ...


@export
def berlekamp_massey(sequence, output="minimal"):
    r"""
    Finds the minimal polynomial $c(x)$ that produces the linear recurrent sequence $y$.

    This function implements the Berlekamp-Massey algorithm.

    Arguments:
        sequence: A linear recurrent sequence $y$ in $\mathrm{GF}(p^m)$.
        output: The output object type.

            - `"minimal"` (default): Returns the minimal polynomial that generates the linear recurrent sequence.
              The minimal polynomial is a characteristic polynomial $c(x)$ of minimal degree.
            - `"fibonacci"`: Returns a Fibonacci LFSR that produces $y$.
            - `"galois"`: Returns a Galois LFSR that produces $y$.

    Returns:
        The minimal polynomial $c(x)$, a Fibonacci LFSR, or a Galois LFSR, depending on the value of `output`.

    Notes:
        The minimal polynomial is the characteristic polynomial $c(x)$ of minimal degree that produces the
        linear recurrent sequence $y$.

        $$c(x) = x^{n} - c_{n-1}x^{n-1} - c_{n-2}x^{n-2} - \dots - c_{1}x - c_{0}$$

        $$y_t = c_{n-1}y_{t-1} + c_{n-2}y_{t-2} + \dots + c_{1}y_{t-n+2} + c_{0}y_{t-n+1}$$

        For a linear sequence with order $n$, at least $2n$ output symbols are required to determine the
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

        Use the Berlekamp-Massey algorithm to return equivalent Fibonacci LFSR that reproduces the sequence.

        .. ipython:: python

            lfsr = galois.berlekamp_massey(y, output="fibonacci")
            print(lfsr)
            z = lfsr.step(y.size); z
            np.array_equal(y, z)

        Use the Berlekamp-Massey algorithm to return equivalent Galois LFSR that reproduces the sequence.

        .. ipython:: python

            lfsr = galois.berlekamp_massey(y, output="galois")
            print(lfsr)
            z = lfsr.step(y.size); z
            np.array_equal(y, z)

    Group:
        linear-sequences
    """
    verify_isinstance(sequence, FieldArray)
    verify_isinstance(output, str)
    if not sequence.ndim == 1:
        raise ValueError(f"Argument 'sequence' must be 1-D, not {sequence.ndim}-D.")
    if not output in ["minimal", "fibonacci", "galois"]:
        raise ValueError(f"Argument 'output' must be in ['minimal', 'fibonacci', 'galois'], not {output!r}.")

    field = type(sequence)
    coeffs = berlekamp_massey_jit(field)(sequence)
    characteristic_poly = Poly(coeffs, field=field)

    if output == "minimal":
        return characteristic_poly

    # The first n outputs are the Fibonacci state reversed
    feedback_poly = characteristic_poly.reverse()
    state_ = sequence[0 : feedback_poly.degree][::-1]
    fibonacci_lfsr = FLFSR(feedback_poly, state=state_)

    if output == "fibonacci":
        return fibonacci_lfsr

    return fibonacci_lfsr.to_galois_lfsr()


class berlekamp_massey_jit(Function):
    """
    Finds the minimal polynomial c(x) of the input sequence.
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
        N = sequence.size
        s = sequence
        c = np.zeros(N, dtype=sequence.dtype)
        b = np.zeros(N, dtype=sequence.dtype)
        c[0] = 1  # The polynomial c(x) = 1
        b[0] = 1  # The polynomial b(x) = 1
        L = 0
        m = 1
        bb = 1

        for n in range(0, N):
            d = 0
            for i in range(0, L + 1):
                d = ADD(d, MULTIPLY(s[n - i], c[i]))

            if d == 0:
                m += 1
            elif 2 * L <= n:
                t = c.copy()
                d_bb = MULTIPLY(d, RECIPROCAL(bb))
                for i in range(m, N):
                    c[i] = SUBTRACT(c[i], MULTIPLY(d_bb, b[i - m]))
                L = n + 1 - L
                b = t.copy()
                bb = d
                m = 1
            else:
                d_bb = MULTIPLY(d, RECIPROCAL(bb))
                for i in range(m, N):
                    c[i] = SUBTRACT(c[i], MULTIPLY(d_bb, b[i - m]))
                m += 1

        return c[0 : L + 1]
