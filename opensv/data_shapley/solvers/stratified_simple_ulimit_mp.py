from typing import Optional, Any

import numpy as np
from tqdm import tqdm, trange
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
        pld: Array=None,
        sub_range: range=None,
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = np.asarray(list(range(N)))
    sh = np.zeros(N)
    
    for i in tqdm(sub_range):
        player, loc = pld[i]
        rng.shuffle(idxes)
        xloc = np.argwhere(idxes == player)
        idxes[xloc], idxes[loc] = idxes[loc], idxes[xloc]
        x_temp, y_temp = x_train[idxes[:loc], :], y_train[idxes[:loc]]
        acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        x_temp, y_temp = x_train[idxes[:loc+1], :], y_train[idxes[:loc+1]]
        new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        xoi = new_acc - acc
        sh[player] += xoi
    
    return sh

def stratified_simple_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))

    rng = np.random.default_rng()
    N = len(y_train)
    T = num_utility // 2
    
    # init_acc = get_utility([], [], x_valid, y_valid, clf)
    # final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    
    sh = np.zeros(N)
    per = T // (N * N)
    pcnt = np.zeros(N)
        
    pld = []    
    for player in trange(N):
        for loc in range(N):
            for _ in range(per):
                pld.append((player, loc))
        pcnt[player] = N * per
    for _ in range(T - per * N * N):
        p = rng.integers(N)
        l = rng.integers(N)
        pld.append((p, l))
        pcnt[p] += 1
        
    sub_length = split_permutation_num(T, num_proc)
    cur = 0
    sub_range = []
    for i in sub_length:
        sub_range.append(range(cur, cur + i))
        cur += i
        
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, pld)
    ret = pool.map(func, sub_range)
    pool.close()
    pool.join()
    
    sh = np.sum(ret, axis=0)
    val = sh / pcnt
    
    return val