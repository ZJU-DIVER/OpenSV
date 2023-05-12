from typing import Optional, Any, Callable, Union

import numpy as np

from ..valuation import Valuation
from ..config import ParamsTable
from solvers import *

Array = np.ndarray

class DataShapley(Valuation):
    def __init__(self):
        self.clf = None
        self.y_valid = None
        self.x_valid = None
        self.y_train = None
        self.x_train = None
        self.num_perm = None
        self.truncated_threshold = None

        self.values = None

    def load(self,
             x_train: Array,
             y_train: Array,
             x_valid: Optional[Array]=None,
             y_valid: Optional[Array]=None,
             clf: Optional[Any]=None,
             para_tbl: Optional[ParamsTable]=None,
             **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.clf = clf
        self.num_perm = para_tbl.num_perm
        self.truncated_threshold = para_tbl.truncated_threshold
        
        # Customized parameters
        self.kwargs = kwargs


    def check_params(self) -> None:
        assert self.num_perm > 0
        assert self.truncated_threshold >= 0

    def solve(self, solver: Optional[Union[str, Callable[..., Array]]]="truncated_mc") -> None:
        self.check_params()
        args = [self.x_train, self.y_train, self.x_valid, self.y_valid, self.clf]
        if isinstance(solver, str):
            match solver:
                case "monte_carlo":
                    self.values = monte_carlo(*args,
                                              self.num_perm)
                case "truncated_mc":
                    self.values = truncated_mc(*args,
                                               self.num_perm,
                                               self.truncated_threshold)
                case _:
                    raise ValueError("[!] No matched solver")
        elif isinstance(solver, Callable[..., Array]):
            # Customize solver
            self.values = solver(*args, **self.kwargs)
        else:
            raise ValueError("[!] Illegal solver type")
        
    def get_values(self) -> Array:
        if self.values is not None:
            return self.values
        else:
            raise ValueError("[!] No values computed")
    