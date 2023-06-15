from typing import Optional, Any
import torch
import numpy as np


Array = np.ndarray

def LSH_sv(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        K: int =16
) -> Array:
    #todo