from typing import Optional, Any

import numpy as np
from tqdm import trange

from opensv.utils.utils import get_utility

Array = np.ndarray

def stratified(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
) -> Array:
    M = num_perm

    N = len(y_train)
    idxes = np.asarray(list(range(N)))

    sh = np.zeros(N * N).reshape((N, N))
    sum_cuad = np.zeros(N)
    mexp = np.ceil(np.max([M / (2 * N * N), 2])).astype(int)

    for player in trange(N):
        for loc in range(N):
            for _ in range(mexp):
                np.random.shuffle(idxes)
                xloc = np.argwhere(idxes == player)
                idxes[xloc], idxes[loc] = idxes[loc], idxes[xloc]

                x_temp, y_temp = x_train[idxes[:loc], :], y_train[idxes[:loc]]
                acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = x_train[idxes[:loc+1], :], y_train[idxes[:loc+1]]
                new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                xoi = new_acc - acc

                sh[player][loc] += xoi
                sum_cuad[loc] += xoi * xoi
            
    s2 = np.zeros(N * N).reshape((N, N))
    for player in trange(N):
        s2[player] = sum_cuad
    s2 = (s2 - sh ** 2 / mexp) / (mexp - 1)
    sum_s2 = np.sum(s2)
    mst = np.ceil(s2 * M / sum_s2 - mexp)
    
    for player in trange(N):
        for loc in range(N):
            m = int(max(0, mst[player][loc]))
            for _ in range(m):
                np.random.shuffle(idxes)
                xloc = np.argwhere(idxes == player)[0][0]
                idxes[xloc], idxes[loc] = idxes[loc], idxes[xloc]

                x_temp, y_temp = x_train[idxes[:loc], :], y_train[idxes[:loc]]
                acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = x_train[idxes[:loc+1], :], y_train[idxes[:loc+1]]
                new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                xoi = new_acc - acc

                sh[player][loc] += xoi
            sh[player][loc] = sh[player][loc] / (m + mexp)
    
    val = np.sum(sh, axis=1)
    return val