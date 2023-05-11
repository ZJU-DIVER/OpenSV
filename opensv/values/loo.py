from typing import Optional, Dict

import numpy as np
from tqdm import trange

from ..utils.utils import get_utility


def loo(x_train, y_train, x_valid, y_valid, clf, params: Optional[Dict] = None):
    """_summary_

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        x_valid (_type_): _description_
        y_valid (_type_): _description_
        clf (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = len(y_train)
    val = np.zeros((N))
    total_idxes = list(range(N))
    acc_in = get_utility(x_train, y_train, x_valid, y_valid, clf)

    for i in trange(N):
        idxes = list(total_idxes).remove(i)  # implict copy
        x_temp, y_temp = x_train[idxes, :], y_train[idxes]
        acc_ex = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        val[i] = acc_in - acc_ex
    return val