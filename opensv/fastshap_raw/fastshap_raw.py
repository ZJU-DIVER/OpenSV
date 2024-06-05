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
from fastshap import FastSHAP
from fastshap import Surrogate, KLDivLoss
from fastshap.utils import MaskLayer1d

# Most params below are from https://github.com/iancovert/fastshap/blob/main/notebooks/census.ipynb
# TODO: Customize those params


class FastShapRaw(Valuation):
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

    def solve(self, model_save_path: str = None) -> None:
        self.check_params()

        ss = StandardScaler()
        ss.fit(self.x_train)
        self.x_train = ss.transform(self.x_train)
        self.x_valid = ss.transform(self.x_valid)

        num_features = self.x_train.shape[1]
        num_class = len(np.unique(self.y_train))

        params = {
            "objective": "multiclass",
            "num_class": num_class,
            "metric": "multi_logloss",
            "verbose": -1,
        }

        d_train = lgb.Dataset(self.x_train, label=self.y_train)
        d_val = lgb.Dataset(self.x_valid, label=self.y_valid)
        model = lgb.train(
            params,
            d_train,
            valid_sets=[d_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(1000),
            ],
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        surr = nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * num_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, num_class),
        ).to(device)

        surrogate = Surrogate(surr, num_features)

        def original_model(x):
            pred = model.predict(x.cpu().numpy())
            return torch.tensor(pred, dtype=torch.float32, device=x.device)

        surrogate.train_original_model(
            self.x_train,
            self.x_valid,
            original_model,
            batch_size=64,
            max_epochs=100,
            loss_fn=KLDivLoss(),
            validation_samples=10,
            validation_batch_size=10000,
            verbose=False,
        )

        explainer = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class * num_features),
        ).to(device)

        fastshap = FastSHAP(explainer, surrogate, link=nn.Softmax(dim=-1))

        fastshap.train(
            self.x_train,
            self.x_valid,
            batch_size=32,
            num_samples=32,
            max_epochs=200,
            validation_samples=128,
            verbose=False,
        )

        val = fastshap.shap_values(self.x_train)
        self.values = [
            np.sum(val[i][:, int(self.y_train[i])]) for i in range(len(self.x_train))
        ]

        if model_save_path is not None:
            torch.save(fastshap.explainer, os.path.join(model_save_path, "explainer.pt"))
            torch.save(fastshap.imputer, os.path.join(model_save_path, "imputer.pt"))

    def get_values(self) -> Array:
        if self.values is not None:
            return self.values
        else:
            raise ValueError("[!] No values computed")
