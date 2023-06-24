from typing import Optional, Any, Callable, Union
from ..utils.volume import *
import numpy as np
from ..utils.utils import clock
from ..valuation import Valuation
from ..config import ParamsTable
from .solvers import *

Array = np.ndarray


'''
    RVShapley only needs a train dataset, and a validation dataset is unnecessary.
    Before using RVShapley, the datasets need to be preprocessed.
    The format of the train dataset is as follows(for example):
        tensor([[x1,x2,x3...,x8],...,[],...,[...]])
        It contains n_participants data subsets ( data owners ),
        each data subsets ( data owners ) consisting of s data points, with each data point having 8 dimensions(x1,x2...x8).

''' 

class RVShapley(Valuation):
    def __init__(self):
        self.clf = None
        self.y_valid = None
        self.x_valid = None
        self.y_train = None
        self.x_train = None
        self.omega = None
        self.values = None
        self.num_perm = None

    def load(self,
             x_train: Array,
             y_train: Optional[Array]=None,
             x_valid: Optional[Array]=None,
             y_valid: Optional[Array]=None,
             clf: Optional[Any]=None,
             para_tbl: Optional[ParamsTable]=ParamsTable(),
             **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.clf = clf
        self.omega = para_tbl.omega
        self.num_perm = para_tbl.num_perm
        
        # Customized parameters
        self.kwargs = kwargs


    def check_params(self) -> None:
        assert self.omega >= 0 and self.omega <= 1

    @clock
    def solve(self, solver: Optional[Union[str, Callable[..., Array]]]="monte_carlo") -> None:
        self.check_params()
        args = [self.x_train , self.omega]
        if isinstance(solver, str):
            match solver:
                case "monte_carlo":
                    self.values = monte_carlo(*args,
                                              self.num_perm)
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
    