from typing import Optional, Any

import numpy as np
from tqdm import tqdm, trange
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
        sub_range: range=None
) -> tuple[Array, Array]:
    N = len(y_train)
    
    sv = []

    for k in tqdm(sub_range):
        idxes = []
        for i in range(N):
            if (k >> i) & 1:
                idxes.append(i)
        
        x_temp, y_temp = x_train[idxes, :], y_train[idxes]
        sv.append(get_utility(x_temp, y_temp, x_valid, y_valid, clf))

    return sv

def subtask2(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        factorial: Array=None,
        utility: Array=None,
        sub_range: range=None
) -> tuple[Array, Array]:
    N = len(y_train)
    
    sv = np.zeros(N)

    for k in tqdm(sub_range):
        sz = 0
        for i in range(N):
            if (k >> i) & 1:
                sz += 1
        coef = factorial[sz] * factorial[N - sz - 1] / factorial[N]
        for i in range(N):
            if not ((k >> i) & 1):
                sv[i] += coef * (utility[k + (1 << i)] - utility[k])

    return sv

from itertools import chain, combinations

def power_set(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def exact_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))


    # n = len(y_train)
    # ext_sv = np.zeros(n)
    # coef = np.zeros(n)
    # fact = np.math.factorial
    # coalition = np.arange(n)
    # for s in trange(n):
    #     coef[s] = fact(s) * fact(n - s - 1) / fact(n)
    # sets = list(power_set(coalition))
    # for idx in trange(len(sets)):
    #     # S = np.zeros((1, n), dtype=bool)
    #     # S[0][list(sets[idx])] = 1
    #     x_temp, y_temp = x_train[list(sets[idx]), :], y_train[list(sets[idx])]
    #     u = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
    #     for i in sets[idx]:
    #         ext_sv[i] += coef[len(sets[idx]) - 1] * u
    #     for i in set(coalition) - set(sets[idx]):
    #         ext_sv[i] -= coef[len(sets[idx])] * u

    # return ext_sv

    N = len(y_train)
    T = 2 ** N
    factorial = np.zeros(N + 1)
    factorial[0] = 1
    for i in range(1, N + 1):
        factorial[i] = factorial[i - 1] * i

    sub_length = split_permutation_num(T, num_proc)
    cur = 0
    sub_range = []
    for i in sub_length:
        sub_range.append(range(cur, cur + i))
        cur += i

    pool = Pool()
    func = partial(subtask1, x_train, y_train, x_valid, y_valid, clf)
    ret = pool.map(func, sub_range)
    pool.close()
    pool.join()

    utility = np.concatenate(ret, axis=0)

    # for i in utility:
    #     print(i, end=", ")

    pool = Pool()
    func = partial(subtask2, x_train, y_train, x_valid, y_valid, clf, factorial, utility)
    ret = pool.map(func, sub_range)
    pool.close()
    pool.join()
    
    val = np.sum(ret, axis=0)
    # print(np.sum(val))

    return val