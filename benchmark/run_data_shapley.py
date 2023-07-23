
import sys
sys.path.append('..')
from opensv import DataShapley
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

from sacred import Experiment
from sacred.observers import MongoObserver



DATABASE_NAME = 'SV_Bench'
YOUR_CPU = None

EXPERIMENT_NAME = 'my_experiment_data-shapley'
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))


@ex.config
def dataset_config():
    dataset_name = 'synthetic'
    train_size = 100
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

    # Step 2: use seed to split then get train and test data
    dv = DataShapley()
    dv.load(train_data[0], train_data[1], valid_data[0], valid_data[1])
    # Step 3: repeat with different seeds and record Shapley values
    for i in range(repeat_time):
        local_seed = seed + i
        np.random.seed(local_seed)
        dv.solve(solver)
        ex.log_scalar(
            f'solver{solver}_perm{num_perms}_{i+1}/{repeat_time}', str(dv.get_values()))


@ex.automain
def main(train_size, valid_size, seed, num_perms, _run):
    feature = np.random.rand(1000, 2)  # X
    label = np.random.choice([0, 1], 1000)  # y

    _, X, _, y = train_test_split(
        feature, label, test_size=train_size + valid_size, random_state=seed)
    train_data = (X[:train_size], y[:train_size])
    valid_data = (X[train_size:], y[train_size:])

    for sol in ['monte_carlo', 'truncated_mc']:
        cal_sv(train_data, valid_data, num_perms, sol)
