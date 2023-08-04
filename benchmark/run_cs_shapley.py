import sys
sys.path.append('..')
from sacred.observers import MongoObserver
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np
import json
from opensv import CSShapley
from dataloader.preparedata_openml import *
DATABASE_NAME = 'SV_Bench'
YOUR_CPU = "localhost:27017"

EXPERIMENT_NAME = 'my_experiment_cs-shapley'
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def dataset_config():
    dataset_name = '2dplanes'
    dataset_id = 727
    train_size = 1000
    valid_size = 100
    seed = 42


@ex.config
def running_config():
    num_perms = 10
    num_procs = 1
    repeat_time = 2


@ex.capture
def cal_sv(train_data: Tuple,
           valid_data: Tuple,
           num_perms: int,
           solver: str,
           seed: int,
           repeat_time: int,
           num_procs: int,  # not used
           _run):

    dv = CSShapley()
    print(train_data[0][:10],train_data[1][:10])
    dv.load(train_data[0], train_data[1], valid_data[0], valid_data[1])
    # Step 3: repeat with different seeds and record Shapley values
    for i in range(repeat_time):
        local_seed = seed + i
        np.random.seed(local_seed)
        dv.solve(solver)
        ex.log_scalar(
            f'solver_{solver}_perm_{num_perms}_{i+1}/{repeat_time}',
            json.dumps(list(dv.get_values()))
        )
        ex.log_scalar(
            f'solver_{solver}_perm_{num_perms}_{i+1}/{repeat_time}_h',
            str(dv.get_values())
        )


@ex.automain
def main(dataset_name, dataset_id, train_size, valid_size, seed, num_perms, _run):
    # step 1: data loader
    # feature = np.random.rand(1000, 2)  # X
    # label = np.random.choice([0, 1], 1000)  # y
    x_train, y_train, x_val, y_val = get_processed_data(dataset_name, dataset_id, train_size, valid_size, seed)

    # Step 2: use seed to split then get train and test data
    # _, X, _, y = train_test_split(
    #     feature, label, test_size=train_size + valid_size, random_state=seed)
    # train_data = (X[:train_size], y[:train_size])
    # valid_data = (X[train_size:], y[train_size:])
    # for sol in ['truncated_mc']:
    #     cal_sv((x_train, y_train), (x_val, y_val), num_perms, sol)
