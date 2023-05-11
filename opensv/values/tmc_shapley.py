from typing import Optional, Dict

import numpy as np
from tqdm import trange

from ..utils.utils import get_utility


def tmc_shapley(x_train, y_train, x_valid, y_valid, clf, params: Optional[Dict] = None):
    # Extract params
    T = params.get('tmc_iter', 20)  # aka the number of permutations
    epsilon = params.get('tmc_thresh', 1e-3)

    N = len(y_train)
    val = np.zeros((N))
    idxes = list(range(N))

    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    for _ in trange(T):
        np.random.shuffle(idxes)
        acc = 0  # init value
        for i in range(1, N + 1):
            if abs(final_acc - acc) < epsilon:
                break
            x_temp, y_temp = x_train[idxes[:j], :], y_train[idxes[:j]]
            new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
            val[idxes[i]] += new_acc - acc
            acc = new_acc
    return val / T
