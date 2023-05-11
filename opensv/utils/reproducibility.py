import random
import numpy as np
import torch

from typing import *
from .logging import logger

def set_seed(seed:Optional[int] = None):
    """set seed for reproducibility

    Args:
        seed (:obj:`int`): the seed to seed everything for reproducibility. if None, do nothing.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Global seed set to {seed}")
