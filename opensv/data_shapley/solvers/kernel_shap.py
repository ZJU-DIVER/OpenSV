from typing import Optional, Any

import numpy as np
from tqdm import trange
from scipy.special import comb

from opensv.utils.utils import get_utility

Array = np.ndarray

def kernel_shap(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
) -> Array:   
    N = len(y_train)
    val = np.zeros(N)
    idxes = list(range(N))
    utilities = np.empty(num_perm)
    coef = np.empty(num_perm)

    coalitons = np.zeros((num_perm,N))
    weight_list = np.array([1 / (cl * (N-cl)) for cl in range(1,N)])
    coef_list = np.array([(N-1) / (comb(N,cl)* cl * (N-cl)) for cl in range(1,N)])

    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    init_acc = get_utility([], [], x_valid, y_valid, clf )

    for _ in range(num_perm):
        np.random.shuffle(idxes)
        i = np.random.choice(a=range(1, N), p=np.array(weight_list) / np.sum(weight_list))
        x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
        utilities[_] = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        coef[_] = coef_list[i-1]
        coalitons[_,idxes[:i]] = 1
    
    A = np.einsum('ij,ik->jk', coalitons, coalitons) / (num_perm/2)
    b = np.sum(coalitons * (utilities - init_acc)[:, np.newaxis], axis=0) / (num_perm/2)
    
    Ainv = np.linalg.inv(A)
    val = Ainv @ (b - np.ones(N)*(np.sum(Ainv @ b) - final_acc + init_acc)/np.sum(Ainv))
    return val
