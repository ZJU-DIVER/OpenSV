from typing import Optional, Any

import numpy as np
from tqdm import trange
from functools import partial
from multiprocessing import Pool

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def reverse(a: Array, n: int) -> Array:
    a_have = set(a)
    na = []
    for i in range(n):
        if i not in a_have:
            na.append(i)
    na = np.asarray(na)
    return na

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
    shp = np.zeros((N, N))
    shm = np.zeros((N, N))
    cp = np.zeros((N, N))
    cm = np.zeros((N, N))

    def update(a: Array) -> None:
        v = 0
        if len(a) > 0:
            v = get_utility(x_train[a, :], y_train[a], x_valid, y_valid, clf)
        else:
            v = get_utility([], [], x_valid, y_valid, clf)
        sz = len(a)
        na = reverse(a, N)
        if sz > 0:
            shp[sz - 1][a] += v
            cp[sz - 1][a] += 1
        if sz < N:
            shm[sz][na] += v
            cm[sz][na] += 1

    for _ in trange(sub_length):
        sz = rng.choice(N + 1, 1)
        A = rng.choice(range(N), sz, replace=False)
        nA = np.asarray(reverse(A, N))
        update(A)
        # update(nA)

    return (shp, shm, cp, cm)

def svarm_stratified_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    
    N = len(y_train)
    T = num_utility
        
    shp = np.zeros((N, N))
    shm = np.zeros((N, N))
    cp = np.zeros((N, N))
    cm = np.zeros((N, N))

    dis = np.zeros(N + 1)
    if N % 2 == 0:
        nlogn = N * np.log(N)
        harmonic = np.sum([1 / i for i in range(1, N // 2)])
        frac = (nlogn - 1) / (2 * nlogn * (harmonic - 1))
        for i in range(2, N // 2):
            dis[i] = frac / i
            dis[N - i] = frac / i
        dis[N // 2] = 1 / nlogn
    else:
        harmonic = np.sum([1 / i for i in range(1, N // 2 + 1)])
        frac = 1 / (2 * (harmonic - 1))
        for i in range(2, N // 2 + 1):
            dis[i] = frac / i
            dis[N - i] = frac / i
    
    probs = [dis[i] for i in range(N + 1)]
    probs = np.asarray(probs)

    def update(a: Array) -> None:
        v = get_utility(x_train[a, :], y_train[a], x_valid, y_valid, clf)
        sz = len(a)
        na = reverse(a, N)
        if sz > 0:
            shp[sz - 1][a] += v
            cp[sz - 1][a] += 1
        if sz < N:
            shm[sz][na] += v
            cm[sz][na] += 1
    
    idxes = list(range(N))
    idxes = np.asarray(idxes)
    update(idxes)
    for i in range(N):
        idxes[i], idxes[N - 1] = idxes[N - 1], idxes[i]
        update(idxes[:N - 1])
        idxes[i], idxes[N - 1] = idxes[N - 1], idxes[i]
        update(idxes[0:1])
    T -= N + N + 1

    for sz in trange(2, N - 1):
        np.random.shuffle(idxes)
        for i in range(0, N // sz):
            A = np.asarray([idxes[i * sz - 1 + j] for j in range(1, sz + 1)])
            v = get_utility(x_train[A, :], y_train[A], x_valid, y_valid, clf)
            T -= 1
            for j in A:
                shp[sz - 1][j] = v
                cp[sz - 1][j] = 1

        if N % sz != 0:
            A = np.asarray([idxes[i - 1] for i in range(N - (N % sz) + 1, N + 1)])
            nA = reverse(A, N)
            B = np.asarray(np.random.choice(nA, sz - (N % sz), replace=False))
            AB = np.concatenate([A, B], axis=0)
            v = get_utility(x_train[AB, :], y_train[AB], x_valid, y_valid, clf)
            T -= 1
            for i in A:
                shp[sz - 1][i] = v
                cp[sz - 1][i] = 1
    
    for sz in trange(2, N - 1):
        np.random.shuffle(idxes)
        for i in range(0, N // sz):
            A = np.asarray([idxes[i * sz - 1 + j] for j in range(1, sz + 1)])
            nA = reverse(A, N)
            v = get_utility(x_train[nA, :], y_train[nA], x_valid, y_valid, clf)
            T -= 1
            for j in nA:
                shm[N - sz][j] = v
                cm[N - sz][j] = 1
        
        if N % sz != 0:
            A = np.asarray([idxes[i - 1] for i in range(N - (N % sz) + 1, N + 1)])
            nA = reverse(A, N)
            B = np.asarray(np.random.choice(nA, sz - (N % sz), replace=False))
            AB = np.concatenate([A, B], axis=0)
            nAB = reverse(AB, N)
            v = get_utility(x_train[nAB, :], y_train[nAB], x_valid, y_valid, clf)
            T -= 1
            for i in A:
                shm[N - sz][i] = v
                cm[N - sz][i] = 1

    # T = T // 2

    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()

    shp2 = np.sum([r[0] for r in ret], axis=0)
    shm2 = np.sum([r[1] for r in ret], axis=0)
    cp2 = np.sum([r[2] for r in ret], axis=0)
    cm2 = np.sum([r[3] for r in ret], axis=0)

    cp = (cp + cp2).clip(min=1)
    cm = (cm + cm2).clip(min=1)

    shp = (shp + shp2) / cp
    shm = (shm + shm2) / cm
    
    val = np.sum((shp - shm) / N, axis=0)
    # val = val * (final_acc - init_acc) / np.sum(val)

    return val