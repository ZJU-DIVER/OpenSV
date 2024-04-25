# Temporary example using existing amortized-valuation.
# License is compatible (MIT)
# Extra requirements here:
# $ pip install git+https://github.com/iancovert/amortized-valuation.git
# $ pip install opendataval lightning

from typing import Optional, Any, Callable, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import lightning as pl
from ..utils.utils import clock
from ..valuation import Valuation
from ..config import ParamsTable
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

Array = np.ndarray

# Me? Rewrite this?
from adv import ValuationModel, Classifier, MarginalContributionDataset
from .utils import SubsetTMCSampler, aggregate_estimator_results

import os
import glob
import tempfile
from copy import deepcopy
from shutil import rmtree
from opendataval.dataval import DataShapley
from opendataval.metrics import Metrics
from opendataval.model import ModelFactory
from opendataval.experiment import ExperimentMediator
from opendataval.dataloader import mix_labels, DataFetcher


class AmortizedValuationRaw(Valuation):
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
        **kwargs,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.kwargs = kwargs

    def check_params(self) -> None:
        return

    def solve(self) -> None:

        self.check_params()

        # Part 1. Make a bunch of initial monte carlo results

        y_train_oh = nn.functional.one_hot(
            torch.from_numpy(self.y_train).to(torch.int64)
        )
        y_valid_oh = nn.functional.one_hot(
            torch.from_numpy(self.y_valid).to(torch.int64)
        )

        def get_experiment_mediator():
            fetcher = DataFetcher.from_data_splits(
                self.x_train,
                y_train_oh.numpy(),
                self.x_valid,
                y_valid_oh.numpy(),
                self.x_train[:100] if len(self.x_train) > 100 else self.x_train,
                (
                    y_train_oh[:100].numpy()
                    if len(y_train_oh.numpy()) > 100
                    else y_train_oh.numpy()
                ),
                one_hot=True,
            )
            pred_model = ModelFactory(model_name="sklogreg", fetcher=fetcher)
            return ExperimentMediator(fetcher=fetcher, pred_model=pred_model)

        temp_path = tempfile.mkdtemp()

        for _ in range(20):
            seed = np.random.randint(2**31)

            exper_med = get_experiment_mediator()
            sampler = SubsetTMCSampler(
                mc_epochs=10,
                random_state=seed,
                cache_name="cached",
            )
            data_evaluator = DataShapley(sampler=sampler)
            data_evaluator.setup(
                exper_med.fetcher, exper_med.pred_model, exper_med.metric
            )

            num_points = len(exper_med.fetcher.x_train)
            relevant_inds = np.arange(num_points)
            data_evaluator.train_data_values(relevant_inds=relevant_inds)

            results = {
                "estimates": sampler.total_contribution / sampler.total_count,
                "total_contribution": sampler.total_contribution,
                "total_count": sampler.total_count,
                "relevant_inds": relevant_inds,
            }
            np.save(os.path.join(temp_path, f"mc-{seed}.npy"), results)
            print(os.path.join(temp_path, f"mc-{seed}.npy"))

        # Part 2. Train a valuation network using results above

        # Classifier pretraining

        batch_size = 32
        exper_med = get_experiment_mediator()
        data_evaluator = DataShapley(
            gr_threshold=1.05,
            max_mc_epochs=300,
            cache_name="cached",
        )
        data_evaluator.setup(exper_med.fetcher, exper_med.pred_model, exper_med.metric)
        x_train_ts = data_evaluator.x_train
        y_train_ts = data_evaluator.y_train  # one-hot

        input_dim = x_train_ts.shape[1]
        output_dim = y_train_ts.shape[1]
        classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        classifier_dataset = TensorDataset(x_train_ts, y_train_ts.argmax(dim=1))
        train_loader = DataLoader(
            classifier_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            classifier_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        model = Classifier(classifier, lr=1e-3)
        trainer = pl.Trainer(
            max_epochs=20,
            precision="bf16-mixed",
            log_every_n_steps=10,
            num_sanity_val_steps=0,
            callbacks=[TQDMProgressBar(refresh_rate=10)],
        )
        trainer.fit(model, train_loader, val_loader)

        # Valuation model training

        lr = 5e-4
        min_lr = 5e-6
        train_contributions = 10
        val_contributions = 10
        target_scaling = 0.001

        filenames = np.sort(glob.glob(os.path.join(temp_path, f"mc-*.npy")))
        results = [
            np.load(filename, allow_pickle=True).item() for filename in filenames
        ]
        print(results)
        train_results = aggregate_estimator_results(results[:train_contributions])
        val_results = aggregate_estimator_results(results[-val_contributions:])

        train_delta = torch.tensor(train_results["estimates"]).unsqueeze(1).float()
        train_dataset = MarginalContributionDataset(x_train_ts, y_train_ts, train_delta)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        val_delta = torch.tensor(val_results["estimates"]).unsqueeze(1).float()
        val_dataset = MarginalContributionDataset(x_train_ts, y_train_ts, val_delta)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # Set up for epochs grid

        best_loss = np.inf
        best_checkpoint = None

        for epochs in [10, 20, 30, 40, 50]:
            classifier_copy = deepcopy(classifier)
            nn.init.zeros_(classifier_copy[-1].weight)
            nn.init.zeros_(classifier_copy[-1].bias)
            model = ValuationModel(
                model=classifier_copy,
                target_scaling=target_scaling,
                lr=lr,
                min_lr=min_lr,
            )

            best_callback = ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                filename="{epoch}-{val_loss:.8f}",
                verbose=True,
            )
            epoch_callback = ModelCheckpoint(every_n_epochs=1, filename="{epoch}")
            trainer = pl.Trainer(
                max_epochs=epochs,
                precision="bf16-mixed",
                log_every_n_steps=10,
                num_sanity_val_steps=0,
                callbacks=[
                    TQDMProgressBar(refresh_rate=10),
                    best_callback,
                    epoch_callback,
                ],
            )
            trainer.fit(model, train_loader, val_loader)

            # Store results.
            loss = best_callback.best_model_score.item()
            checkpoint = best_callback.best_model_path
            if loss < best_loss:
                best_loss = loss
                best_checkpoint = checkpoint

        # Part 3. Evaluate the trained checkpoint

        checkpoint = ValuationModel.load_from_checkpoint(best_checkpoint)
        self.values = checkpoint.estimates.numpy()

        rmtree(temp_path, ignore_errors=True)

    def get_values(self) -> Array:
        if self.values is not None:
            return self.values
        else:
            raise ValueError("[!] No values computed")
