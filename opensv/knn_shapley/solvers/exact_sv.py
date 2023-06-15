from typing import Optional, Any
import torch
import numpy as np


Array = np.ndarray

def exact_sv(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        K: int =16
) -> Array:

    x_train=torch.from_numpy(x_train).type(torch.float32)
    y_train=torch.from_numpy(y_train).type(torch.float32)
    x_valid=torch.from_numpy(x_valid).type(torch.float32)
    y_valid=torch.from_numpy(y_valid).type(torch.float32)

    N = len(x_train)
    M = len(x_valid)

    dist = torch.cdist(x_train.view(len(x_train), -1), x_valid.view(len(x_valid), -1))
    _, indices = torch.sort(dist, axis=0)
    y_sorted = y_train[indices]

    score = torch.zeros_like(dist)

    score[indices[N-1], range(M)] = (y_sorted[N-1] == y_valid).float() / N
    for i in range(N-2, -1, -1):
        score[indices[i], range(M)] = score[indices[i+1], range(M)] + \
                                    1/K * ((y_sorted[i] == y_valid).float() - (y_sorted[i+1] == y_valid).float()) * min(K, i+1) / (i+1)
    return score.mean(axis=1).numpy()