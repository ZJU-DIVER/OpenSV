import sys
sys.path.append('..')
# from sacred.observers import MongoObserver
# from sacred import Experiment
# from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split
from typing import Callable
import numpy as np
import json
from opensv import DivisiveShapley
from dataloader.preparedata_openml import *
import time
# DATABASE_NAME = 'SV_Bench'
# YOUR_CPU = "localhost:27017"

# EXPERIMENT_NAME = 'my_experiment_divisive-shapley'
# ex = Experiment(EXPERIMENT_NAME)
# ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))
# ex.captured_out_filter = apply_backspaces_and_linefeeds


# def dataset_config():
#     dataset_name = '2dplanes'
#     dataset_id = 727
#     train_size = 1000
#     valid_size = 100
#     seed = 42


# def running_config():
#     num_perms = 10
#     num_procs = 1
#     repeat_time = 2


def cal_sv(x_data: np.array,
           solver: Callable[[np.ndarray], float],
           seed: int,
           num_procs: int,  # not used
           _run):

    dv = DivisiveShapley(beta=2)
    t0 = time.time()
    values = dv.calculate(x_data,solver)
    print(values)
    print("time:{:.5f}s".format(time.time() - t0))
    # ex.log_scalar(
    #     f'solver_{solver}_perm_{num_perms}_{i+1}/{repeat_time}',
    #     json.dumps(list(dv.get_values()))
    # )
    # ex.log_scalar(
    #     f'solver_{solver}_perm_{num_perms}_{i+1}/{repeat_time}_h',
    #     str(dv.get_values())
    # )


def characteristic_function(S: np.ndarray) -> float:
    # Placeholder for characteristic function
    return np.sum(S, axis=1)  # Example: sum of values


if __name__ == "__main__":
    dataset_name = '2dplanes'
    dataset_id = 727
    seed = 42

    num_procs = 1

    x_train, _, _, _ = get_processed_data(dataset_name, dataset_id, 1000, 0, seed)

    cal_sv(x_train.reshape(-1,1), characteristic_function, seed, num_procs, "divisive_shapey")