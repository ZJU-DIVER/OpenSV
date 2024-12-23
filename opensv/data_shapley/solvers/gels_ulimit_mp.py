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
        q: Array = None,
        sub_length: int=0
) -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    r = np.zeros((N))
    t = np.zeros((N))
    idxes = np.asarray(list(range(N)))

    for _ in trange(sub_length):
        s = rng.choice(range(1, N), p=q)
        rng.shuffle(idxes)
        x_temp, y_temp = x_train[idxes[:s], :], y_train[idxes[:s]]
        u = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        t[idxes[:s]] += 1
        r[idxes[:s]] += u

    return (r, t)

def gels_ulimit_mp(
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
    T = num_utility - 2

    q = np.zeros((N - 1))
    for i in range(1, N):
        q[i - 1] = N / (i * (N - i))
    q = q / np.sum(q)
    
    r = np.zeros((N))
    t = np.zeros((N))
        
    sub_length = split_permutation_num(T, num_proc)

    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, q)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()

    r = np.sum([r[0] for r in ret], axis=0)
    t = np.sum([r[1] for r in ret], axis=0)
    t = t.clip(min=1)
    r /= t

    har = np.sum([1.00 / i for i in range(1, N)])
    r *= har
    base = (final_acc - init_acc - np.sum(r)) / N
    val = r + base
    
    return val