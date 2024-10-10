from typing import Optional, Any

import numpy as np
from tqdm import trange
from functools import partial
from multiprocessing import Pool

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def subtask(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        final_acc: float=1,
        epsilon: float=0,
        sub_length: int=0
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))
    val = np.zeros(N)
    for _ in trange(sub_length):
        rng.shuffle(idxes)
        acc = 0
        for i in range(1, N + 1):
            if abs(final_acc - acc) < epsilon:
                break
            x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
            new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
            val[idxes[i-1]] += new_acc - acc
            acc = new_acc
    return val

def truncated_mc_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        truncated_threshold: float=0.001,
        num_proc: int=1,
) -> Array:
    T = num_perm
    epsilon = truncated_threshold

    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, final_acc, epsilon)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    ret_val = np.asarray(ret)
    val = ret_val.sum(axis=0) / T

    return val