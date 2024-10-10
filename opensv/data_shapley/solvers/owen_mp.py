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
        num_q_split: int=100,
        sub_length: int=0
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    val = np.zeros(N)
    for _ in trange(sub_length):
        for q in range(num_q_split):    
            prob = 1.00 * q / (num_q_split - 1)
            for i in range(N):
                p_list = np.array(rng.binomial(1, prob, N))
                p_list[i] = 0
                chosen = np.nonzero(p_list)[0]
                x_temp, y_temp = x_train[chosen, :], y_train[chosen]
                prev_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = np.concatenate([x_temp, [x_train[i]]]), np.concatenate([y_temp, [y_train[i]]])
                cur_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                val[i] += cur_acc - prev_acc
    return val

def owen_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        num_q_split: int=100,
        num_proc: int=1,
) -> Array:
    T = num_perm

    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, num_q_split)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    ret_val = np.asarray(ret)
    val = ret_val.sum(axis=0) / (T * num_q_split)

    return val