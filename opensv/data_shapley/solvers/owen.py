from typing import Optional, Any

import numpy as np
from tqdm import trange

from opensv.utils.utils import get_utility

Array = np.ndarray

def owen(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        num_q_split: int=100,
) -> Array:
    T = num_perm
    N = len(y_train)
    val = np.zeros(N)
    for _ in trange(T):
        for q in trange(num_q_split):    
            prob = 1.00 * q / (num_q_split - 1)
            for i in range(N):
                p_list = np.array(np.random.binomial(1, prob, N))
                p_list[i] = 0
                chosen = np.nonzero(p_list)[0]
                x_temp, y_temp = x_train[chosen, :], y_train[chosen]
                prev_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                x_temp, y_temp = np.concatenate([x_temp, [x_train[i]]]), np.concatenate([y_temp, [y_train[i]]])
                cur_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
                val[i] += cur_acc - prev_acc
    
    return val / (T * num_q_split)
