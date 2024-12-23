from typing import Optional, Any

import numpy as np
from tqdm import trange
from functools import partial
from multiprocessing import Pool

from scipy.optimize import minimize
from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def subtask(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        q: Array=None,
        sub_length: int=None
) -> tuple[Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))

    idxes = np.asarray(list(range(N)))
    b = np.zeros(N * sub_length).reshape((N, sub_length))
    u = np.zeros(sub_length)

    for t in trange(sub_length):
        length = rng.choice(range(1, N), p=q)
        rng.shuffle(idxes)
        for i in range(length):
            b[idxes[i]][t] = 1
        x_temp, y_temp = x_train[idxes[:length], :], y_train[idxes[:length]]
        u[t] = get_utility(x_temp, y_temp, x_valid, y_valid, clf)

    return (b, u)

def group_testing_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
        truncated_threshold: float=0.00001,
        solver_iter: int=50000,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    
    T = num_utility - 2

    N = len(y_train)
    Z = 0
    for k in range(1, N):
        Z += 1.00 / k
    Z *= 2
    q = np.zeros(N - 1)
    for k in range(1, N):
        q[k - 1] = 1.00 / k + 1.00 / (N - k)
    q = q / np.sum(q)
    print(q)
    
    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, q)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    
    b = np.concatenate([r[0] for r in ret], axis=1)
    u = np.concatenate([r[1] for r in ret], axis=0)

    delta_u = np.zeros(N * N).reshape((N, N))
    for i in range(N):
        for j in range(N):
            delta_u[i][j] = np.sum((b[i] - b[j]) * u) * Z
            delta_u[i][j] /= np.max([np.sum(np.abs(b[i] - b[j])), 1])

    print(delta_u)

    un = final_acc - init_acc
    val = np.zeros(N)
    for i in range(N):
        val[i] = un
        for j in range(N):
            val[i] += delta_u[i][j]
    val /= N
    # val = val * (final_acc - init_acc) / np.sum(val)

    return val
