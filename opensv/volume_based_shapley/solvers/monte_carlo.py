from typing import Optional, Any
import torch
import numpy as np
from tqdm import trange
from opensv.utils.volume import *

Array = np.ndarray

def monte_carlo(
        x_train: Array,
        y_train: Optional[Array] = None,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        omega: Optional[Any] = None,
        num_perm: int=500,
) -> Array:
    T = num_perm
    D = x_train.size()[2]
    N = len(x_train)
    val = torch.zeros(N)
    idxes = list(range(N))

    if(omega):
        for _ in trange(num_perm):
            np.random.shuffle(idxes)
            prefix_vol = 0  # init value
            for i in range(1, N + 1):
                x_temp = x_train[idxes[:i], :].reshape(-1,D)
                X_tilde, cubes = compute_X_tilde_and_counts(x_temp, omega)
                curr_vol = compute_robust_volumes([X_tilde], [cubes])[0]
                val[idxes[i-1]] += curr_vol - prefix_vol
                prefix_vol = curr_vol
        return (val / T).numpy()
    else: 
        for _ in trange(num_perm):
            np.random.shuffle(idxes)
            prefix_vol = 0  # init value
            for i in range(1, N + 1):
                x_temp = x_train[idxes[:i], :].reshape(-1,D)
                curr_vol = torch.sqrt(torch.linalg.det(x_temp.T @ x_temp) + 1e-8)
                val[idxes[i-1]] += curr_vol - prefix_vol
                prefix_vol = curr_vol
        return (val / T).numpy()