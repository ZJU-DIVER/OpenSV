import sys
sys.path.append('..')
from sacred.observers import FileStorageObserver
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import Callable
import numpy as np
import json
from opensv import DivisiveShapley
from dataloader.preparedata_openml import *
import time


DATABASE_NAME = 'SV_Bench'
YOUR_CPU = None

EXPERIMENT_NAME = 'my_experiment_divisive-shapley'
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver.create("../results"))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def dataset_config():
    dataset_name = '2dplanes'
    dataset_id = 727
    train_size = 10000
    valid_size = 1000
    seed = 42


@ex.config
def running_config():
    # num_perms = 10
    num_procs = 1
    repeat_time = 2


@ex.capture
def cal_sv(x_data: np.array,
           solver: Callable[[np.ndarray], float],
           repeat_num: int,
           repeat_time: int,
           seed: int,
           num_procs: int  # not used
           ):

    dv = DivisiveShapley(beta=1000 ** (1 / np.sqrt(1000)))
    # t0 = time.time()
    values = dv.calculate(x_data,solver)
    # print(values)
    # print("time:{:.5f}s".format(time.time() - t0))
    ex.log_scalar(
        f'solver_{solver}_reapeat_{repeat_num}/{repeat_time}',
        str(values)
    )
    # ex.log_scalar(
    #     f'solver_{solver}_perm_{num_perms}_{i+1}/{repeat_time}_h',
    #     str(values)
    # )


@ex.automain
def main(train_size, valid_size, seed, dataset_name, dataset_id, repeat_time):

    for i in range(repeat_time):

        x_train, y_train, x_val, y_val = get_processed_data(dataset_name, dataset_id, train_size, valid_size, seed + i)

        model = LogisticRegression()

        model.fit(x_train, y_train)

        cal_sv(x_train, model, i + 1)   