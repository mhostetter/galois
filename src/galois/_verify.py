"""
A module containing various input verification functions.
"""

from __future__ import annotations

import builtins
import inspect
from typing import Any

import numpy as np
import numpy.typing as npt


def _argument_names():
    """
    Finds the source code argument names from the function that called a verification function.
    """
    frame = inspect.currentframe()
    if frame is None:
        return ("<argument>",)

    outer_frames = inspect.getouterframes(frame)
    if len(outer_frames) < 3:
        return ("<argument>",)

    frame_info = inspect.getframeinfo(outer_frames[2].frame)  # function() -> verify() -> _argument_name()
    if not frame_info.code_context:
        return ("<argument>",)

    string = "".join(frame_info.code_context).strip()
    start = string.find("(")
    end = string.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return ("<argument>",)

    args = string[start + 1 : end].split(",")
    args = [arg.strip() for arg in args]  # Strip leading/trailing whitespace
    # args = [arg.split("=")[0].strip() for arg in args]  # Remove default values and strip whitespace
    return tuple(args)


def verify_positional_args(args: tuple[Any], limit: int):
    """
    Verify a limited number of positional arguments.
    """
    if len(args) > limit:
        raise ValueError(f"A max of {limit} positional arguments are acceptable, not {_argument_names()}.")

    return len(args)


def verify_specified(arg: Any) -> Any:
    """
    Verifies that the argument is not None.
    """
    if arg is None:
        raise ValueError(f"Argument {_argument_names()[0]!r} must be provided, not {arg}.")

    return arg


def verify_not_specified(arg: Any) -> Any:
    """
    Verifies that the argument is None.
    """
    if arg is not None:
        raise ValueError(f"Argument {_argument_names()[0]!r} must not be provided, not {arg}.")

    return arg


def verify_only_one_specified(*args):
    """
    Verifies that only one of the arguments is not None.
    """
    if sum(arg is not None for arg in args) != 1:
        raise ValueError(f"Exactly one of the arguments {_argument_names()} must be provided, not {args}.")

    return args


def verify_at_least_one_specified(*args):
    """
    Verifies that at least one of the arguments is not None.
    """
    if all(arg is None for arg in args):
        raise ValueError(f"At least one of the arguments {_argument_names()} must be provided, not {args}.")

    return args


def verify_isinstance(
    arg: Any,
    types: Any,
    optional: bool = False,
) -> Any:
    """
    Verifies that the argument is an instance of the specified type(s).
    """
    if optional:
        # TODO: Can this be done in a more elegant way?
        try:
            types = list(types)
        except TypeError:
            types = [types]
        types = tuple(types + [type(None)])

    if not isinstance(arg, types):
        raise TypeError(f"Argument {_argument_names()[0]!r} must be an instance of {types}, not {type(arg)}.")

    return arg


def verify_issubclass(argument, types, optional=False):
    """
    Verifies that the argument is a subclass of the type(s).
    """
    if optional and argument is None:
        return argument

    # Need this try/except because issubclass(instance, (classes,)) will itself raise a TypeError.
    # Instead, we'd like to raise our own TypeError.
    try:
        valid = issubclass(argument, types)
    except TypeError:
        valid = False

    if not valid:
        raise TypeError(f"Argument {_argument_names()[0]!r} must be a subclass of {types}, not {type(argument)}.")


def verify_same_types(*args):
    """
    Verifies that all arguments are of the same type.
    """
    first_type = type(args[0])
    for i, arg in enumerate(args[1:], start=1):
        if not isinstance(arg, first_type):
            raise TypeError(
                f"Arguments {_argument_names()} must be of the same type, but argument {_argument_names()[i]!r} is of type {type(arg)!r} while argument {_argument_names()[0]!r} is of type {first_type!r}."
            )


def verify_literal(
    arg: Any,
    literals: Any,
):
    """
    Verifies that the argument is one of the specified literals.
    """
    if not arg in literals:
        raise ValueError(f"Argument {_argument_names()[0]!r} must be one of {literals}, not {arg!r}.")

    return arg


def verify_scalar(
    x: Any,
    # Data types
    optional: bool = False,
    int: bool = False,
    float: bool = False,
    complex: bool = False,
    # Value constraints
    real: bool = False,
    imaginary: bool = False,
    negative: bool = False,
    non_negative: bool = False,
    positive: bool = False,
    even: bool = False,
    odd: bool = False,
    power_of_two: bool = False,
    inclusive_min: float | None = None,
    inclusive_max: float | None = None,
    exclusive_min: float | None = None,
    exclusive_max: float | None = None,
    # Conversions
    accept_numpy: bool = True,
    convert_numpy: bool = False,
) -> Any:
    """
    Verifies that the argument is a scalar and satisfies the conditions.
    """
    if optional and x is None:
        return x

    if convert_numpy:
        x = convert_to_scalar(x)

    if isinstance(x, np.ndarray) and x.ndim > 0:
        raise TypeError(f"Argument {_argument_names()[0]!r} must be a scalar, not an array.")

    is_integer = isinstance(x, builtins.int) or (accept_numpy and isinstance(x, np.integer))

    if int:
        if not is_integer:
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int, not {type(x)}.")
    if float:
        if not (isinstance(x, (builtins.int, builtins.float)) or (accept_numpy and np.issubdtype(x, np.floating))):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float, not {type(x)}.")
    if complex:
        if not (
            isinstance(x, (builtins.int, builtins.float, builtins.complex))
            or (accept_numpy and np.issubdtype(x, np.complexfloating))
        ):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float or complex, not {type(x)}.")

    if real:
        if np.iscomplexobj(x) and not np.isrealobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be real, not complex.")
    if imaginary:
        if not np.iscomplexobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be complex, not real.")

    if negative:
        if x >= 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be negative, not {x}.")
    if non_negative:
        if x < 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be non-negative, not {x}.")
    if positive:
        if x <= 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be positive, not {x}.")
    if even:
        if not is_integer:
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int to be even, not {type(x)}.")
        if x % 2 != 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be even, not {x}.")
    if odd:
        if not is_integer:
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int to be odd, not {type(x)}.")
        if x % 2 == 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be odd, not {x}.")
    if power_of_two:
        if not is_integer:
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int to be a power of two, not {type(x)}.")
        if x <= 0 or not (x & (x - 1) == 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be a power of two, not {x}.")

    if inclusive_min is not None:
        if x < inclusive_min:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at least {inclusive_min}, not {x}.")
    if inclusive_max is not None:
        if x > inclusive_max:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at most {inclusive_max}, not {x}.")
    if exclusive_min is not None:
        if x <= exclusive_min:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be greater than {exclusive_min}, not {x}.")
    if exclusive_max is not None:
        if x >= exclusive_max:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be less than {exclusive_max}, not {x}.")

    return x


def verify_bool(
    x: Any,
    # Conversions
    accept_numpy: bool = True,
    convert_numpy: bool = False,
) -> bool:
    """
    Verifies that the argument is a boolean.
    """
    if convert_numpy:
        x = convert_to_scalar(x)

    if not (isinstance(x, bool) or (accept_numpy and isinstance(x, np.bool_))):
        raise TypeError(f"Argument {_argument_names()[0]!r} must be a bool, not {type(x)}.")

    return x


def verify_coprime(
    x: int,
    y: int,
):
    """
    Verifies that the arguments are coprime.
    """
    if np.gcd(x, y) != 1:
        raise ValueError(
            f"Arguments {_argument_names()[0]!r} and {_argument_names()[1]!r} must be coprime, not {x} and {y}."
        )


def verify_condition(
    condition: bool,
):
    """
    Verifies that the condition is satisfied.
    """
    if not condition:
        raise ValueError(f"Arguments must satisfy the condition {_argument_names()[0]!r}.")


def verify_arraylike(
    x: npt.ArrayLike | None,
    dtype: npt.DTypeLike | None = None,
    # Data types
    optional: bool = False,
    int: bool = False,
    float: bool = False,
    complex: bool = False,
    # Value constraints
    real: bool = False,
    imaginary: bool = False,
    negative: bool = False,
    non_negative: bool = False,
    positive: bool = False,
    inclusive_min: float | None = None,
    inclusive_max: float | None = None,
    exclusive_min: float | None = None,
    exclusive_max: float | None = None,
    # Dimension and size constraints
    atleast_1d: bool = False,
    atleast_2d: bool = False,
    atleast_3d: bool = False,
    ndim: int | None = None,
    size: int | None = None,
    sizes: tuple | list | None = None,
    size_multiple: int | None = None,
    shape: tuple[int, ...] | None = None,
    square: bool | None = None,
) -> npt.NDArray | None:
    """
    Converts the argument to a NumPy array and verifies the conditions.
    """
    if optional and x is None:
        return x

    x = np.asarray(x, dtype=dtype)

    x = verify_ndarray(
        x,
        dtype=dtype,
        # Data types
        int=int,
        float=float,
        complex=complex,
        # Value constraints
        real=real,
        imaginary=imaginary,
        negative=negative,
        non_negative=non_negative,
        positive=positive,
        inclusive_min=inclusive_min,
        inclusive_max=inclusive_max,
        exclusive_min=exclusive_min,
        exclusive_max=exclusive_max,
        # Dimension and size constraints
        atleast_1d=atleast_1d,
        atleast_2d=atleast_2d,
        atleast_3d=atleast_3d,
        ndim=ndim,
        size=size,
        sizes=sizes,
        size_multiple=size_multiple,
        shape=shape,
        square=square,
    )

    return x


def verify_ndarray(
    x: npt.NDArray,
    # Array type
    subclass: type = np.ndarray,
    dtype: npt.DTypeLike | None = None,
    # Data types
    int: bool = False,
    float: bool = False,
    complex: bool = False,
    # Value constraints
    real: bool = False,
    imaginary: bool = False,
    negative: bool = False,
    non_negative: bool = False,
    positive: bool = False,
    inclusive_min: float | None = None,
    inclusive_max: float | None = None,
    exclusive_min: float | None = None,
    exclusive_max: float | None = None,
    # Dimension and size constraints
    atleast_1d: bool = False,
    atleast_2d: bool = False,
    atleast_3d: bool = False,
    ndim: int | None = None,
    size: int | None = None,
    sizes: tuple | list | None = None,
    size_multiple: int | None = None,
    shape: tuple[int, ...] | None = None,
    square: bool | None = None,
):
    """
    Verifies that the argument is a NumPy ndarray and satisfies the conditions.
    """
    verify_isinstance(x, subclass)

    if dtype:
        if x.dtype != np.dtype(dtype):
            raise TypeError(f"Argument {_argument_names()[0]!r} must have dtype {dtype}, not {x.dtype}.")

    if int:
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int, not {x.dtype}.")
    if float:
        if not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float, not {x.dtype}.")
    if complex:
        if not (
            np.issubdtype(x.dtype, np.integer)
            or np.issubdtype(x.dtype, np.floating)
            or np.issubdtype(x.dtype, np.complexfloating)
        ):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float or complex, not {x.dtype}.")

    if real:
        if not np.isrealobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be real, not complex.")
    if imaginary:
        if not np.iscomplexobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be complex, not real.")
    if negative:
        if np.any(x >= 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be negative, not {x}.")
    if non_negative:
        if np.any(x < 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be non-negative, not {x}.")
    if positive:
        if np.any(x <= 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be positive, not {x}.")

    if inclusive_min is not None:
        if np.any(x < inclusive_min):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at least {inclusive_min}, not {x}.")
    if inclusive_max is not None:
        if np.any(x > inclusive_max):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at most {inclusive_max}, not {x}.")
    if exclusive_min is not None:
        if np.any(x <= exclusive_min):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be greater than {exclusive_min}, not {x}.")
    if exclusive_max is not None:
        if np.any(x >= exclusive_max):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be less than {exclusive_max}, not {x}.")

    if atleast_1d:
        x = np.atleast_1d(x)
    if atleast_2d:
        x = np.atleast_2d(x)
    if atleast_3d:
        x = np.atleast_3d(x)
    if ndim is not None:
        if not x.ndim == ndim:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have {ndim} dimensions, not {x.ndim}.")
    if size is not None:
        if not x.size == size:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have {size} elements, not {x.size}.")
    if sizes is not None:
        if not x.size in sizes:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have one of {sizes} elements, not {x.size}.")
    if size_multiple is not None:
        if not x.size % size_multiple == 0:
            raise ValueError(
                f"Argument {_argument_names()[0]!r} must have a size that is a multiple of {size_multiple}, not {x.size}."
            )
    if shape is not None:
        if not x.shape == shape:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have shape {shape}, not {x.shape}.")
    if square is not None:
        if square and not x.shape[0] == x.shape[1]:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be square, not {x.shape}.")
        if not square and x.shape[0] == x.shape[1]:
            raise ValueError(f"Argument {_argument_names()[0]!r} must not be square, not {x.shape}.")

    return x


def verify_same_shape(
    x: npt.NDArray,
    y: npt.NDArray,
):
    """
    Verifies that the arguments have the same shape.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"Arguments {_argument_names()[0]!r} and {_argument_names()[1]!r} must have the same shape, not {x.shape} and {y.shape}."
        )


def convert_to_scalar(x: Any):
    """
    Converts the input to a scalar if possible.
    """
    if np.isscalar(x) and hasattr(x, "item"):
        x = x.item()

    # TODO: Why is this needed? array(0) with np.int64 does not return true for np.isscalar()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    return x


def convert_output(
    x: Any,
    squeeze: bool = False,
) -> Any:
    """
    Converts the output to a native Python type if scalar.
    """
    if squeeze:
        x = np.squeeze(x)

    x = convert_to_scalar(x)

    return x
