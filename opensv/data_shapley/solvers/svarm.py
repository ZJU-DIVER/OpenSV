from typing import Optional, Any

import numpy as np
from tqdm import trange

from opensv.utils.utils import get_utility

Array = np.ndarray

harmonic = 0

def sample_plus(n: int) -> Array:
    global harmonic
    sz = np.random.choice(range(1, n + 1), 1, p=[1.00 / (i * harmonic) for i in range(1, n + 1)])
    return np.random.choice(list(range(n)), sz, replace=False)

def sample_minus(n: int) -> Array:
    global harmonic
    sz = np.random.choice(range(0, n), 1, p=[1.00 / ((n - i) * harmonic) for i in range(0, n)])
    return np.random.choice(list(range(n)), sz, replace=False)


def svarm(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
) -> Array:
    N = len(y_train)
    T = num_perm - 2 * (N + 1)
    global harmonic
    harmonic = np.sum([1.00 / i for i in range(1, N + 1)])

    shp = np.zeros(N)
    shm = np.zeros(N)
    cp = np.ones(N)
    cm = np.ones(N)

    for i in range(N):
        idx = []
        for j in range(N):
            if i != j:
                idx.append(j)
        idx = np.asarray(idx)

        sz = np.random.choice(N, 1)
        selected = np.random.choice(idx, sz, replace=False)
        shp[i] = get_utility(x_train[selected, :], y_train[selected], x_valid, y_valid, clf)

        sz = np.random.choice(N, 1)
        selected = np.random.choice(idx, sz, replace=False)
        shm[i] = get_utility(x_train[selected, :], y_train[selected], x_valid, y_valid, clf)

    for _ in range(T):
        ap = sample_plus(N)
        am = sample_minus(N)
        am_have = set(am)
        nam = []
        for i in range(N):
            if i not in am_have:
                nam.append(i)
        nam = np.asarray(nam)

        x_temp, y_temp = x_train[ap, :], y_train[ap]
        vp = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        x_temp, y_temp = x_train[am, :], y_train[am]
        vm = get_utility(x_temp, y_temp, x_valid, y_valid, clf)

        # shp[ap] = (shp[ap] * cp[ap] + vp) / (cp[ap] + 1)
        shp[ap] += vp
        cp[ap] += 1
        # shm[nam] = (shm[nam] * cm[nam] + vm) / (cm[nam] + 1)
        shm[nam] += vm
        cm[nam] += 1

    shp = shp / cp
    shm = shm / cm
    val = shp - shm
    return val
