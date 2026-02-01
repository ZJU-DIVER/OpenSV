from typing import Optional, Any

import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Manager

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray


def get_utility_memorized(
    mem, x: Array, y: Array, index: Array, x_valid: Array, y_valid: Array, clf
):
    # n = len(y)
    # if len(index) > 20 and len(index) < n - 20:
    #     x_temp, y_temp = x[index, :], y[index]
    #     return get_utility(x_temp, y_temp, x_valid, y_valid, clf), True

    id = 0
    for i in index:
        id |= 1 << i
    if id in mem:
        mem[-1] += 1
        return mem[id], True
        # return mem[id], False
    x_temp, y_temp = x[index, :], y[index]
    u = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
    mem[id] = u
    return u, True


def subtask(
    mem,
    x_train: Array,
    y_train: Array,
    x_valid: Optional[Array] = None,
    y_valid: Optional[Array] = None,
    clf: Optional[Any] = None,
    init_acc: float = 0,
    final_acc: float = 1,
    num_utility: int = 0,
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))
    val = np.zeros(N)
    cnt = np.zeros(N)
    with tqdm(total=num_utility) as pbar:
        while num_utility > 0:
            rng.shuffle(idxes)
            acc = init_acc
            for i in range(1, N + 1):
                new_acc, used = get_utility_memorized(
                    mem, x_train, y_train, idxes[:i], x_valid, y_valid, clf
                )
                if used:
                    num_utility -= 1
                    pbar.update(1)
                val[idxes[i - 1]] += new_acc - acc
                cnt[idxes[i - 1]] += 1
                acc = new_acc
            if num_utility <= 0:
                break
    return val, cnt


def monte_carlo_mem_ulimit_mp(
    x_train: Array,
    y_train: Array,
    x_valid: Optional[Array] = None,
    y_valid: Optional[Array] = None,
    clf: Optional[Any] = None,
    num_proc: int = 1,
    num_utility: int = 500,
) -> Array:
    print(
        get_utility(x_train, y_train, x_valid, y_valid, clf),
        get_utility([], [], x_valid, y_valid, clf),
    )

    T = num_utility - 2
    N = len(y_train)

    global cost_after_find
    cost_after_find = mem_param[2]
    mem = hybrid_dict()
    mem.init(N, mem_param[0], mem_param[1])
    mem._set(-1, 0)

    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    print(final_acc - init_acc)
    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(
        subtask, mem, x_train, y_train, x_valid, y_valid, clf, init_acc, final_acc
    )
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()

    sv = np.sum([r[0] for r in ret], axis=0)
    cnt = np.sum([r[1] for r in ret], axis=0)
    cnt = cnt.clip(min=1)
    val = sv / cnt
    print(cnt, np.sum(cnt))

    # ret_val = np.asarray(ret)
    # val = ret_val.sum(axis=0) / T * N
    # val = val * (final_acc - init_acc) / np.sum(val)

    print(f"Duplicated count: {mem._get(-1)} / {num_utility}")
    with open("log.txt", "a") as f:
        f.write(f"monte_carlo_mem_ulimit_mp,{N},{num_utility},{mem._get(-1)}\n")

    return val
