from typing import Optional, Any

import numpy as np
from tqdm import trange
from functools import partial
from multiprocessing import Pool

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

harmonic = 0

def sample_plus(n: int, rng) -> Array:
    global harmonic
    sz = rng.choice(range(1, n + 1), 1, p=[1.00 / (i * harmonic) for i in range(1, n + 1)])
    return rng.choice(list(range(n)), sz, replace=False)

def sample_minus(n: int, rng) -> Array:
    global harmonic
    sz = rng.choice(range(0, n), 1, p=[1.00 / ((n - i) * harmonic) for i in range(0, n)])
    return rng.choice(list(range(n)), sz, replace=False)

def subtask(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        sub_length: int=0
) -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    shp = np.zeros(N)
    shm = np.zeros(N)
    cp = np.zeros(N)
    cm = np.zeros(N)

    for _ in trange(sub_length):
        ap = sample_plus(N, rng)
        am = sample_minus(N, rng)
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

        shp[ap] += vp
        cp[ap] += 1
        shm[nam] += vm
        cm[nam] += 1

    return (shp, shm, cp, cm)

def svarm_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        num_proc: int=1,
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

    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    
    shp += np.sum([r[0] for r in ret], axis=0)
    shm += np.sum([r[1] for r in ret], axis=0)
    cp += np.sum([r[2] for r in ret], axis=0)
    cm += np.sum([r[3] for r in ret], axis=0)

    shp = shp / cp
    shm = shm / cm
    val = shp - shm

    return val