from typing import Optional, Any

import numpy as np
from tqdm import trange
from functools import partial
from multiprocessing import Pool

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def subtask1(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        mexp: int=0,
        sub_players: range=None,
) -> tuple[Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = np.asarray(list(range(N)))

    sh = np.zeros(N * N).reshape((N, N))
    sum_cuad = np.zeros(N)
    for player in sub_players:
        for loc in range(N):
            for _ in range(mexp):
                rng.shuffle(idxes)
                xloc = np.argwhere(idxes == player)
                idxes[xloc], idxes[loc] = idxes[loc], idxes[xloc]

                x_temp, y_temp = x_train[idxes[:loc], :], y_train[idxes[:loc]]
                acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = x_train[idxes[:loc+1], :], y_train[idxes[:loc+1]]
                new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                xoi = new_acc - acc

                sh[player][loc] += xoi
                sum_cuad[loc] += xoi * xoi
    
    return (sh, sum_cuad)

def subtask2(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        mexp: int=0,
        mst: Array=None,
        old_sh: Array=None,
        sub_players: range=None,
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = np.asarray(list(range(N)))

    sh = np.zeros(N * N).reshape((N, N))
    for player in sub_players:
        for loc in range(N):
            m = int(max(0, mst[player][loc]))
            for _ in range(m):
                rng.shuffle(idxes)
                xloc = np.argwhere(idxes == player)[0][0]
                idxes[xloc], idxes[loc] = idxes[loc], idxes[xloc]

                x_temp, y_temp = x_train[idxes[:loc], :], y_train[idxes[:loc]]
                acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = x_train[idxes[:loc+1], :], y_train[idxes[:loc+1]]
                new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                xoi = new_acc - acc

                sh[player][loc] += xoi
            sh[player][loc] = (old_sh[player][loc] + sh[player][loc]) / (m + mexp)

    return sh

def stratified_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        num_proc: int=1,
) -> Array:
    M = num_perm
    N = len(y_train)

    sh = np.zeros(N * N).reshape((N, N))
    sum_cuad = np.zeros(N)
    mexp = np.ceil(np.max([M / (2 * N * N), 2])).astype(int)

    sub_length = split_permutation_num(N, num_proc)
    cur = 0
    sub_players = []
    for i in sub_length:
        sub_players.append(range(cur, cur + i))
        cur += i

    pool = Pool()
    func = partial(subtask1, x_train, y_train, x_valid, y_valid, clf, mexp)
    ret = pool.map(func, sub_players)
    pool.close()
    pool.join()

    sh = np.sum([r[0] for r in ret], axis=0)
    sum_cuad = np.sum([r[1] for r in ret], axis=0)

    s2 = np.zeros(N * N).reshape((N, N))
    for player in trange(N):
        s2[player] = sum_cuad
    s2 = (s2 - sh ** 2 / mexp) / (mexp - 1)
    sum_s2 = np.sum(s2)
    mst = np.ceil(s2 * M / sum_s2 - mexp)

    pool = Pool()
    func = partial(subtask2, x_train, y_train, x_valid, y_valid, clf, mexp, mst, sh)
    ret = pool.map(func, sub_players)
    pool.close()
    pool.join()
    ret_val = np.asarray(ret)

    sh = ret_val.sum(axis=0)
    val = np.sum(sh, axis=1)

    return val