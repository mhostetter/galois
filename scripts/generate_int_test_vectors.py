"""
Script to generate unit test vectors for number theoretic functions.

* `sudo apt install sagemath`
"""
import json
import os
import pickle
import random
import shutil

import sage
import numpy as np
from sage.all import xgcd, lcm, prod, isqrt

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
FOLDER = os.path.join(PATH, "data")
if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER)
os.mkdir(FOLDER)

SEED = 123456789


def set_seed(seed):
    """Update the RNG seeds so the LUT is reproducible"""
    np.random.seed(seed)
    random.seed(seed)


def save_pickle(d, folder, name):
    with open(os.path.join(folder, name), "wb") as f:
        pickle.dump(d, f)


###############################################################################
# Math functions
###############################################################################

set_seed(SEED + 101)
X = [random.randint(-1000, 1000) for _ in range(20)] + [random.randint(-1000, 1_000_000_000) for _ in range(20)]
Y = [random.randint(-1000, 1000) for _ in range(20)] + [random.randint(-1000, 1_000_000_000) for _ in range(20)]
D = [0,]*len(X)
S = [0,]*len(X)
T = [0,]*len(X)
for i in range(len(X)):
    x = X[i]
    y = Y[i]
    d, s, t = xgcd(x, y)
    D[i] = int(d)
    S[i] = int(s)
    T[i] = int(t)
d = {"X": X, "Y": Y, "D": D, "S": S, "T": T}
save_pickle(d, FOLDER, "egcd.pkl")

set_seed(SEED + 102)
X = [[random.randint(-1000, 1000) for _ in range(random.randint(2, 6))] for _ in range(20)] + [[random.randint(-1000, 1_000_000) for _ in range(random.randint(2, 6))] for _ in range(20)]
Z = [0,]*len(X)
for i in range(len(X)):
    x = X[i]
    z = lcm(x)
    Z[i] = int(z)
d = {"X": X, "Z": Z}
save_pickle(d, FOLDER, "lcm.pkl")

set_seed(SEED + 103)
X = [[random.randint(-1000, 1000) for _ in range(random.randint(2, 6))] for _ in range(20)] + [[random.randint(-1000, 1_000_000) for _ in range(random.randint(2, 6))] for _ in range(20)]
Z = [0,]*len(X)
for i in range(len(X)):
    x = X[i]
    z = prod(x)
    Z[i] = int(z)
d = {"X": X, "Z": Z}
save_pickle(d, FOLDER, "prod.pkl")

set_seed(SEED + 104)
X = [random.randint(-1000, 1000) for _ in range(20)] + [random.randint(-1000, 1_000_000) for _ in range(20)]
E = [random.randint(0, 1_000) for _ in range(40)]
M = [random.randint(-1000, 1000) for _ in range(20)] + [random.randint(-1000, 1_000_000) for _ in range(20)]
Z = [0,]*len(X)
for i in range(len(X)):
    x = X[i]
    e = E[i]
    m = M[i]
    z = pow(x, e, m)
    Z[i] = int(z)
d = {"X": X, "E": E, "M": M, "Z": Z}
save_pickle(d, FOLDER, "power.pkl")

set_seed(SEED + 105)
X = [random.randint(0, 1000) for _ in range(20)] + [random.randint(1000, 1_000_000_000) for _ in range(20)]
Z = [0,]*len(X)
for i in range(len(X)):
    x = X[i]
    z = isqrt(x)
    Z[i] = int(z)
d = {"X": X, "Z": Z}
save_pickle(d, FOLDER, "isqrt.pkl")
