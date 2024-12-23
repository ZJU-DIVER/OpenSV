# Temporary example using existing fastshap.
# License is compatible (MIT)
# Extra requirements here:
# $ pip install git+https://github.com/iancovert/fastshap.git
# $ pip install lightgbm

from typing import Optional, Any, Callable, Union

import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
import os
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler

from ..valuation import Valuation
from ..config import ParamsTable
from ..utils.utils import clock

Array = np.ndarray

# Me? Rewrite this?
from .betashap.ShapEngine import ShapEngine


class BetaShapleyRaw(Valuation):
    def __init__(self):
        self.y_valid = None
        self.x_valid = None
        self.y_train = None
        self.x_train = None

        self.values = None

    def load(
        self,
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        **kwargs
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.kwargs = kwargs

    def check_params(self) -> None:
        return

    def solve(self, weight, model_family='logistic', metric='f1', GR_threshold=1.05) -> None:
        shap_engine = ShapEngine(X=self.x_train, y=self.y_train, X_val=self.x_valid, y_val=self.y_valid, problem='classification', model_family=model_family, metric=metric, GR_threshold=GR_threshold)
        shap_engine.run(weights_list=[weight])
        self.values = shap_engine.results[f'Beta({weight[0]},{weight[1]})']

    def get_values(self) -> Array:
        if self.values is not None:
            return self.values
        else:
            raise ValueError("[!] No values computed")
