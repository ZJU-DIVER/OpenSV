import abc
from typing import Any, Callable, Optional, Union

import numpy as np

from .config import ParamsTable

Array = np.ndarray


class Valuation(abc.ABC):
    """Base valuation class with constructor and public methods.
    """
    
    @abc.abstractmethod
    def load(self,
             x_train: Array, 
             y_train: Array, 
             x_valid: Optional[Array]=None, 
             y_valid: Optional[Array]=None,
             clf: Optional[Any]=None,
             para_tbl: Optional[ParamsTable]=None,
             **kwargs):
        """load data as the players and models for computing Shapley-based 
        values.
        """
        pass
    
    @abc.abstractmethod
    def check_params(self) -> None:
        """Check whether necessary params for solver are ready
        """
        
    @abc.abstractmethod
    def solve(self, solver: Optional[Union[str, Callable[..., Array]]]=None) -> None:
        """Solve the value computation by various impl.
        """
    
    @abc.abstractmethod
    def get_values(self) -> Array:
        """Return the value results.

        Returns:
            Array: values for each player.
        """