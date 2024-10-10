from typing import Optional, Any

import numpy as np
from tqdm import trange

from opensv.utils.utils import get_utility

Array = np.ndarray

def complementary(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
) -> Array:
    T = num_perm

    N = len(y_train)
    idxes = np.asarray(list(range(N)))

    sv = np.zeros(N * N).reshape((N, N))
    cnt = np.zeros(N * N).reshape((N, N)).astype(int)
    for k in trange(T):
        np.random.shuffle(idxes)
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
    
    cnt = cnt.clip(min=1)
    sum_sv = sv / cnt / N
    val = np.sum(sum_sv, axis=1)
    return val
