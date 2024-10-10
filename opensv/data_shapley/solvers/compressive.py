from typing import Optional, Any

import numpy as np
from tqdm import trange

from opensv.utils.utils import get_utility

Array = np.ndarray

def compressive(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        num_measure: int=500,
) -> Array:
    T = num_perm
    N = len(y_train)
    M = num_measure

    idxes = np.asarray(list(range(N)))


