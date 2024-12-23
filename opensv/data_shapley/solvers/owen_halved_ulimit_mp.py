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
            prob = 0.50 * q / (num_q_split - 1)
            for i in range(N):
                p_list = np.array(rng.binomial(1, prob, N))
                p_list[i] = 0
                chosen = np.nonzero(p_list)[0]
                p_list[i] = 1
                rev_chosen = np.where(p_list == 0)[0]

                x_temp, y_temp = x_train[chosen, :], y_train[chosen]
                prev_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = np.concatenate([x_temp, [x_train[i]]]), np.concatenate([y_temp, [y_train[i]]])
                cur_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                
                x_temp, y_temp = x_train[rev_chosen, :], y_train[rev_chosen]
                rev_prev_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = np.concatenate([x_temp, [x_train[i]]]), np.concatenate([y_temp, [y_train[i]]])
                rev_cur_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                
                val[i] += cur_acc - prev_acc + rev_cur_acc - rev_prev_acc
    
    return val

def owen_halved_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
        num_q_split: int=50,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    
    N = len(y_train)
    T = num_utility // (num_q_split * N * 4)
    if T <= 0:
        num_q_split = num_utility // (N * 4)
        T = num_utility // (num_q_split * N * 4)

    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, num_q_split)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    ret_val = np.asarray(ret)
    val = ret_val.sum(axis=0) / (T * (num_q_split + 1) * 2)
    # val = val * (final_acc - init_acc) / np.sum(val)

    return val