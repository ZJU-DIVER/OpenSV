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
        sub_range: range=None
) -> tuple[Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))
    sv = np.zeros(N * N).reshape((N, N))
    cnt = np.zeros(N * N).reshape((N, N)).astype(int)

    for k in tqdm(sub_range):
        rng.shuffle(idxes)
        i = rng.choice(N + 1, 1)[0]
        x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
        acc_l = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        x_temp, y_temp = x_train[idxes[i:], :], y_train[idxes[i:]]
        acc_r = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        u = acc_l - acc_r
        for j in range(i):
            sv[idxes[j]][i - 1] += u
            cnt[idxes[j]][i - 1] += 1
        for j in range(i, N):
            sv[idxes[j]][N - i - 1] -= u
            cnt[idxes[j]][N - i - 1] += 1
    return (sv, cnt)

def complementary_warmup_ulimit_mp(
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

    sv = np.zeros(N * N).reshape((N, N))
    cnt = np.zeros(N * N).reshape((N, N)).astype(int)
    rng = np.random.default_rng()
    idxes = list(range(N))

    c = -1
    lasm = 0
    while c != lasm:
        c = lasm
        for i in range(N):
            for j in range(N):
                if cnt[i][j] < 2:
                    rng.shuffle(idxes)
                    x_temp, y_temp = x_train[idxes[:j+1], :], y_train[idxes[:j+1]]
                    acc_l = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                    x_temp, y_temp = x_train[idxes[j+1:], :], y_train[idxes[j+1:]]
                    acc_r = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                    num_utility -= 2
                    u = acc_l - acc_r
                    for k in range(j + 1):
                        sv[idxes[k]][j] += u
                        cnt[idxes[k]][j] += 1
                    for k in range(j + 1, N):
                        sv[idxes[k]][N - j - 2] -= u
                        cnt[idxes[k]][N - j - 2] += 1
                    
        lasm = np.sum(cnt[0])

    T = num_utility // 2

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
    
    sv = sv + np.sum([r[0] for r in ret], axis=0)
    cnt = cnt + np.sum([r[1] for r in ret], axis=0)
    cnt = cnt.clip(min=1)
    sum_sv = sv / cnt / N
    val = np.sum(sum_sv, axis=1)
    # val = val * (final_acc - init_acc) / np.sum(val)

    return val