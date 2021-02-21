"""
A module with LUTs for the GF2 class.

Octave:
    clear;
    pkg load communications;

    q = 2;  % Galois field order
    m = 1;  % Galois field prime power GF(2^m)

    % For addition, subtraction, and multiplication
    x = 0:q-1;
    y = 0:q-1;
    X1 = gf(repmat(x', 1, length(y)), m);
    Y1 = gf(repmat(y, length(x), 1), m);

    % For division (y can't be zero)
    x = 0:q-1;
    y = 1:q-1;
    X2 = gf(repmat(x', 1, length(y)), m);
    Y2 = gf(repmat(y, length(x), 1), m);

    % For other operations
    x = 0:q-1;
    X = gf(x, m);

    ADD = X1 .+ Y1;
    SUB = X1 .- Y1;
    MUL = X1 .* Y1;
    DIV = X2 ./ Y2;

    NEG = -X;
    SQR = X .^ 2;
    PWR_3 = X .^ 3;
    PWR_8 = X .^ 8;
    LOG = glog(X);  % Ignore first output

    % Convert back to integers for displaying, copying, and pasting in the Octave GUI
    X1 = X1.x;
    Y1 = Y1.x;
    X2 = X2.x;
    Y2 = Y2.x;
    X = X.x;
    ADD = ADD.x;
    SUB = SUB.x;
    MUL = MUL.x;
    DIV = DIV.x;
    NEG = NEG.x;
    SQR = SQR.x;
    PWR_3 = PWR_3.x;
    PWR_8 = PWR_8.x;
    LOG = LOG.x;
"""
import numpy as np

import galois


X1 = galois.GF2([
    [0,0],
    [1,1],
])

Y1 = galois.GF2([
    [0,1],
    [0,1],
])

X2 = galois.GF2([
    [0,],
    [1,],
])

Y2 = galois.GF2([
    [1,],
    [1,],
])

X = galois.GF2([0,1])

ADD = galois.GF2([
    [0,1],
    [1,0],
])

SUB = galois.GF2([
    [0,1],
    [1,0],
])

MUL = galois.GF2([
    [0,0],
    [0,1],
])

DIV = galois.GF2([
    [0,],
    [1,],
])

NEG = galois.GF2([0,1])

SQR = galois.GF2([0,1])

PWR_3 = galois.GF2([0,1])

PWR_8 = galois.GF2([0,1])

LOG = np.array([-np.Inf,0], dtype=np.float32)
