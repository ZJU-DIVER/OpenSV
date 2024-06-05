import sys

sys.path.append("..")

from opendataval.dataloader import DataFetcher

# from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split

from typing import Tuple
import numpy as np
import json
import time

from opensv import FastShapRaw
from opensv.utils.utils import Float32Encoder

Array = np.ndarray

DATABASE_NAME = "SV_Bench"
YOUR_CPU = None

EXPERIMENT_NAME = "my_experiment_fastshap"
ex = Experiment(EXPERIMENT_NAME)
# ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))
ex.observers.append(FileStorageObserver.create(EXPERIMENT_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def dataset_config():
    dataset_list = ["electricity", "fried", "2dplanes", "pol", "MiniBooNE", "nomao"]
    train_size = 1000
    valid_size = 100
    seed = 42


@ex.config
def running_config():
    num_procs = 1


@ex.capture
def cal_sv(
    dataset_name: str,
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
):
    shap = FastShapRaw()
    shap.load(x_train, y_train, x_val, y_val)
    start_time = time.perf_counter_ns()
    shap.solve()
    end_time = time.perf_counter_ns()
    ex.log_scalar(
        f"fastshap_{dataset_name}/shap_values",
        json.dumps(list(shap.get_values()), cls=Float32Encoder),
    )
    ex.log_scalar(
        f"fastshap_{dataset_name}/time_used",
        end_time - start_time,
    )


def onehot_to_location(y: Array) -> Array:
    res = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(len(y[i])):
            if y[i, j] > 0:
                res[i] = j
                break
    return res


@ex.automain
def main(train_size, valid_size, seed, dataset_list):
    for dataset in dataset_list:
        print(f"Testing {dataset}")
        fetcher = DataFetcher(dataset_name=dataset).split_dataset_by_count(
            train_size, valid_size
        )
        x_train, y_train, x_val, y_val, _, _ = fetcher.datapoints
        x_train = np.array(x_train.tolist())
        y_train = onehot_to_location(np.array(y_train.tolist()))
        x_val = np.array(x_val.tolist())
        y_val = onehot_to_location(np.array(y_val.tolist()))
        cal_sv(dataset, x_train, y_train, x_val, y_val)
