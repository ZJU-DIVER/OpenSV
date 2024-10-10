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
        sub_range: range=None
) -> tuple[Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))
    sv = np.zeros(N * N).reshape((N, N))
    cnt = np.zeros(N * N).reshape((N, N)).astype(int)

    for k in sub_range:
        rng.shuffle(idxes)
        i = k % N
        x_temp, y_temp = x_train[idxes[:i+1], :], y_train[idxes[:i+1]]
        acc_l = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        x_temp, y_temp = x_train[idxes[i+1:], :], y_train[idxes[i+1:]]
        acc_r = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        u = acc_l - acc_r
        for j in range(i):
            sv[idxes[j]][i] += u
            cnt[idxes[j]][i] += 1
        for j in range(i, N):
            sv[idxes[j]][N - i - 1] -= u
            cnt[idxes[j]][N - i - 1] += 1

    return (sv, cnt)

def complementary_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        num_proc: int=1,
) -> Array:
    T = num_perm
    N = len(y_train)

    sv = np.zeros(N * N).reshape((N, N))
    cnt = np.zeros(N * N).reshape((N, N)).astype(int)

    sub_length = split_permutation_num(T, num_proc)
    cur = 0
    sub_range = []
    for i in sub_length:
        sub_range.append(range(cur, cur + i))
        cur += i

    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf)
    ret = pool.map(func, sub_range)
    pool.close()
    pool.join()
    
    sv = np.sum([r[0] for r in ret], axis=0)
    cnt = np.sum([r[1] for r in ret], axis=0)
    cnt = cnt.clip(min=1)
    sum_sv = sv / cnt / N
    val = np.sum(sum_sv, axis=1)

    return val