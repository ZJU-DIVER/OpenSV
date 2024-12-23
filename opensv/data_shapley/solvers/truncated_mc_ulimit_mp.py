from typing import Optional, Any

import numpy as np
from tqdm import tqdm
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
        init_acc: float=0,
        final_acc: float=1,
        epsilon: float=0,
        num_utility: int=0
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))
    val = np.zeros(N)
    perm = 0
    with tqdm(total=num_utility) as pbar:
        while num_utility > 0:
            rng.shuffle(idxes)
            acc = init_acc
            perm += 1
            for i in range(1, N + 1):
                if abs(final_acc - acc) < epsilon:
                    break
                x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
                new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                num_utility -= 1
                pbar.update(1)
                val[idxes[i-1]] += new_acc - acc
                acc = new_acc
                if num_utility <= 0:
                    break
    return (val, perm)

def truncated_mc_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
        truncated_threshold: float=0.001,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    
    T = num_utility - 2
    N = len(y_train)
    epsilon = truncated_threshold

    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, init_acc, final_acc, epsilon)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()

    ret_val = np.sum([r[0] for r in ret], axis=0)
    ret_perm = np.sum([r[1] for r in ret], axis=-1)
    # val = ret_val / ret_perm
    val = ret_val
    val = val * (final_acc - init_acc) / np.sum(val)

    return val