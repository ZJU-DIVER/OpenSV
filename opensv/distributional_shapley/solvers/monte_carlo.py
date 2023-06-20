from typing import Optional, Any

import numpy as np
from tqdm import trange

from opensv.utils.utils import get_utility

Array = np.ndarray

def monte_carlo(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        x_tot: Optional[Array] = None,
        y_tot: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=2000,
) -> Array:
    T = num_perm
    
    train_num = len(y_train)
    tot_num = len(y_tot)
    val = np.zeros(train_num)

    for _ in trange(num_perm):
        idxes = list(tot_num)
        np.random.shuffle(idxes)
        k = np.random.randint(tot_num)
        if k == 0:
            # x_init and y_init are empty
            x_init, y_init = np.empty((0, x_tot.shape[1])), np.empty((0, y_tot.shape[1]))
            init_acc = 0
        else:
            x_init, y_init = x_tot[idxes[:k]], y_tot[idxes[:k]]
            init_acc = get_utility(x_init, y_init, x_valid, y_valid, clf)
        
        for i in range(train_num):
            x_batch = np.concatenate((x_init, x_train[i]))
            y_batch = np.concatenate((y_init, y_train[i]))
            acc = get_utility(x_batch, y_batch, x_valid, y_valid, clf)
            val[i] += acc - init_acc

    return val / T

