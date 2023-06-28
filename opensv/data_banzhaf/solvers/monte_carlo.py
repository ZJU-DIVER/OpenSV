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
        clf: Optional[Any] = None,
        num_perm: int=500,
) -> Array:
    T = num_perm

    N = len(y_train)
    val = np.zeros(N)
    idxes = list(range(N))

    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    for _ in trange(num_perm):
        np.random.shuffle(idxes)
        acc = 0  # init value
        for i in range(1, N + 1):
            if abs(final_acc - acc):
                break
            x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
            new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
            val[idxes[i-1]] += new_acc - acc
            acc = new_acc
    return val / T