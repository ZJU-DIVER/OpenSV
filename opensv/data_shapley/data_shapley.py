from typing import Optional, Any, Callable, Union

import numpy as np
from sklearn.linear_model import LogisticRegression as LR

from ..valuation import Valuation
from ..config import ParamsTable
from .solvers import *
from ..utils.utils import clock

Array = np.ndarray

class DataShapley(Valuation):
    def __init__(self):
        self.clf = None
        self.y_valid = None
        self.x_valid = None
        self.y_train = None
        self.x_train = None
        self.num_perm = None
        self.num_q_split = None
        self.truncated_threshold = None

        self.values = None

    def load(self,
             x_train: Array,
             y_train: Array,
             x_valid: Optional[Array]=None,
             y_valid: Optional[Array]=None,
             clf: Optional[Any]=LR(max_iter=100),
             para_tbl: Optional[ParamsTable]=ParamsTable(),
             **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.clf = clf
        self.num_proc = para_tbl.num_proc
        self.num_perm = para_tbl.num_perm
        self.num_q_split = para_tbl.num_q_split
        self.truncated_threshold = para_tbl.truncated_threshold
        
        # Customized parameters
        self.kwargs = kwargs


    def check_params(self) -> None:
        assert self.num_perm > 0
        assert self.truncated_threshold >= 0

    @clock
    def solve(self, solver: Optional[Union[str, Callable[..., Array]]]="truncated_mc") -> None:
        self.check_params()
        args = [self.x_train, self.y_train, self.x_valid, self.y_valid, self.clf]
        if isinstance(solver, str):
            match solver:
                case "monte_carlo":
                    self.values = monte_carlo(*args,
                                              self.num_perm)
                case "monte_carlo_mp":
                    self.values = monte_carlo_mp(*args,
                                              self.num_perm, self.num_proc)
                case "truncated_mc":
                    self.values = truncated_mc(*args,
                                               self.num_perm,
                                               self.truncated_threshold)
                case "truncated_mc_mp":
                    self.values = truncated_mc_mp(*args,
                                               self.num_perm,
                                               self.truncated_threshold, self.num_proc)
                case "monte_carlo_antithetic":
                    self.values = monte_carlo_antithetic(*args,
                                              self.num_perm)
                case "monte_carlo_antithetic_mp":
                    self.values = monte_carlo_antithetic_mp(*args,
                                              self.num_perm, self.truncated_threshold,self.num_proc)
                case "owen":
                    self.values = owen(*args, self.num_perm, self.num_q_split)
                case "owen_mp":
                    self.values = owen_mp(*args, self.num_perm, self.num_q_split, self.num_proc)
                case "owen_halved":
                    self.values = owen_halved(*args, self.num_perm, self.num_q_split)
                case "owen_halved_mp":
                    self.values = owen_halved_mp(*args, self.num_perm, self.num_q_split, self.num_proc)
                case "stratified":
                    self.values = stratified(*args, self.num_perm)
                case "stratified_mp":
                    self.values = stratified_mp(*args, self.num_perm, self.num_proc)
                case "complementary":
                    self.values = complementary(*args, self.num_perm)
                case "complementary_mp":
                    self.values = complementary_mp(*args, self.num_perm, self.num_proc)
                case "svarm":
                    self.values = svarm(*args, self.num_perm)
                case "svarm_mp":
                    self.values = svarm_mp(*args, self.num_perm, self.num_proc)
                case "svarm_stratified":
                    self.values = svarm_stratified(*args, self.num_perm)
                case "svarm_stratified_mp":
                    self.values = svarm_stratified_mp(*args, self.num_perm, self.num_proc)
                case "kernel_shap":
                    self.values = kernel_shap(*args,
                                              self.num_perm)
                case "kernel_shap_mp":
                    self.values = kernel_shap_mp(*args,
                                              self.num_perm, self.num_proc)
                case "improved_kernel_shap":
                    self.values = improved_kernel_shap(*args,
                                              self.num_perm)
                case "improved_kernel_shap_mp":
                    self.values = improved_kernel_shap_mp(*args,
                                              self.num_perm, self.num_proc)
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
    