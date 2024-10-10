from typing import Optional, Any

import numpy as np
from tqdm import trange
from functools import partial
from multiprocessing import Pool
from scipy.special import comb

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def subtask(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        weight_list: Array=None,
        sub_length: int=0
) -> tuple[Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))

    utilities = np.empty(sub_length)
    coalitons = np.zeros((sub_length, N))
    
    for _ in trange(sub_length):
        rng.shuffle(idxes)
        i = rng.choice(a=range(1, N), p=np.array(weight_list) / np.sum(weight_list))
        x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
        utilities[_] = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        coalitons[_, idxes[:i]] = 1
    return (utilities, coalitons)


def kernel_shap_mp(
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

    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    init_acc = get_utility([], [], x_valid, y_valid, clf)
    weight_list = np.array([1 / (cl * (N-cl)) for cl in range(1,N)])

    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, weight_list)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    
    utilities = np.concatenate([r[0] for r in ret], axis=0)
    coalitons = np.concatenate([r[1] for r in ret], axis=0)
    
    A = np.einsum('ij,ik->jk', coalitons, coalitons) / (num_perm/2)
    b = np.sum(coalitons * (utilities - init_acc)[:, np.newaxis], axis=0) / (num_perm/2)
    
    Ainv = np.linalg.inv(A)
    val = Ainv @ (b - np.ones(N)*(np.sum(Ainv @ b) - final_acc + init_acc)/np.sum(Ainv))

    return val