def add(x, y):
    """
    Adds two Galois field arrays element-wise.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.add.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        y = GF.Random(10); y
        np.add(x, y)
        x + y
    """
    return


def subtract(x, y):
    """
    Subtracts two Galois field arrays element-wise.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.subtract.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        y = GF.Random(10); y
        np.subtract(x, y)
        x - y
    """
    return


def multiply(x, y):
    """
    Multiplies two Galois field arrays element-wise.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.multiply.html

    Examples
    --------
    Multiplying two Galois field arrays results in field multiplication.

    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        y = GF.Random(10); y
        np.multiply(x, y)
        x * y

    Multiplying a Galois field array with an integer results in scalar multiplication.

    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        np.multiply(x, 3)
        x * 3

    .. ipython:: python

        print(GF.properties)
        # Adding `characteristic` copies of any element always results in zero
        x * GF.characteristic
    """
    return


def divide(x, y):
    """
    Divides two Galois field arrays element-wise.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.divide.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        y = GF.Random(10, low=1); y
        z = np.divide(x, y); z
        y * z

    .. ipython:: python

        np.true_divide(x, y)
        x / y
        np.floor_divide(x, y)
        x // y
    """
    return


def negative(x):
    """
    Returns the element-wise additive inverse of a Galois field array.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.negative.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        y = np.negative(x); y
        x + y

    .. ipython:: python

        -x
        -1*x
    """
    return


def reciprocal(x):
    """
    Returns the element-wise multiplicative inverse of a Galois field array.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(5, low=1); x
        y = np.reciprocal(x); y
        x * y

    .. ipython:: python

        x ** -1
        GF(1) / x
        GF(1) // x
    """
    return


def power(x, y):
    """
    Exponentiates a Galois field array element-wise.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.power.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        np.power(x, 3)
        x ** 3
        x * x * x

    .. ipython:: python

        x = GF.Random(10, low=1); x
        y = np.random.randint(-10, 10, 10); y
        np.power(x, y)
        x ** y
    """
    return


def square(x):
    """
    Squares a Galois field array element-wise.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.square.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x = GF.Random(10); x
        np.square(x)
        x ** 2
        x * x
    """
    return


def log(x):
    """
    Computes the logarithm (base `GF.primitive_element`) of a Galois field array element-wise.

    Calling :func:`np.log` implicitly uses base :obj:`galois.FieldMeta.primitive_element`. See
    :func:`galois.FieldArray.log` for logarithm with arbitrary base.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.log.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        alpha = GF.primitive_element; alpha
        x = GF.Random(10, low=1); x
        y = np.log(x); y
        alpha ** y
    """
    return


def matmul(x1, x2):
    """
    Computes the matrix multiplication of two Galois field arrays.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.log.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        x1 = GF.Random((3,4)); x1
        x2 = GF.Random((4,5)); x2
        np.matmul(x1, x2)
        x1 @ x2
    """
    return
