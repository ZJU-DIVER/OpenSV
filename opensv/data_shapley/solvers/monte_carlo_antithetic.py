from typing import Optional, Any

import numpy as np
from tqdm import trange

from opensv.utils.utils import get_utility

Array = np.ndarray

def monte_carlo_antithetic(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        truncated_threshold: float=0
) -> Array:
    T = (num_perm + 1) // 2
    epsilon = truncated_threshold

    N = len(y_train)
    val = np.zeros(N)
    idxes = list(range(N))

    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    for _ in trange(T):
        np.random.shuffle(idxes)
        acc = 0
        rev_acc = final_acc
        for i in range(1, N + 1):
            if abs(final_acc - acc) <= epsilon:
                break
            x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
            x_rev_temp, y_rev_temp = x_train[idxes[i:], :], y_train[idxes[i:]]
            new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
            new_rev_acc = get_utility(x_rev_temp, y_rev_temp, x_valid, y_valid, clf)
            val[idxes[i-1]] += new_acc - acc + rev_acc - new_rev_acc
            acc = new_acc
            rev_acc = new_rev_acc
    return val / (T + T)