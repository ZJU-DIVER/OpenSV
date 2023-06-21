from typing import Optional, Any, Callable, Union

import numpy as np

from ..valuation import Valuation
from ..config import ParamsTable
from .solvers import *

Array = np.ndarray

class KNNShapley(Valuation):
    def __init__(self):
        self.clf = None
        self.y_valid = None
        self.x_valid = None
        self.y_train = None
        self.x_train = None
        self.K = None
        self.eps = None
        self.values = None

    def load(self,
             x_train: Array,
             y_train: Array,
             x_valid: Optional[Array]=None,
             y_valid: Optional[Array]=None,
             clf: Optional[Any]=None,
             para_tbl: Optional[ParamsTable]=ParamsTable(),
             **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.K = para_tbl.K
        self.eps = para_tbl.eps
        
        # Customized parameters
        self.kwargs = kwargs


    def check_params(self) -> None:
        assert self.K > 0

    @clock
    def solve(self, solver: Optional[Union[str, Callable[..., Array]]]="exact_sv") -> None:
        self.check_params()
        args = [self.x_train, self.y_train, self.x_valid, self.y_valid,self.K]
        if isinstance(solver, str):
            match solver:
                case "exact_sv":
                    self.values = exact_sv(*args)
                case "LSH_sv":
                    self.values = LSH_sv(*args)
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
    