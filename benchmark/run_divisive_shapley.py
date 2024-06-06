import sys
sys.path.append('..')
from sacred.observers import FileStorageObserver
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
from opensv import DivisiveShapley
import time
from opensv.games import *
from opensv.utils.utils import Float32Encoder
from opendataval.dataloader import DataFetcher


DATABASE_NAME = 'SV_Bench'
YOUR_CPU = None

EXPERIMENT_NAME = 'my_experiment_divisive-shapley'
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver.create("../results"))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def dataset_config():
    datasets = ["electricity", "fried", "2dplanes", "pol", "MiniBooNE", "nomao"]
    train_size = 1000
    valid_size = 200
    seed = 42


@ex.config
def running_config():
    repeat_time = 1000


@ex.capture
def cal_sv(game: BaseGame,
           repeat_num: int,
           repeat_time: int,
           dataset_name:str
           ):

    dv = DivisiveShapley(beta=1000 ** (1 / np.sqrt(1000)))
    t0 = time.perf_counter_ns()
    values = dv.calculate(game)
    t1 = time.perf_counter_ns()
    ex.log_scalar(
        f'divisive_shap_{dataset_name}_reapeat_{repeat_num}/{repeat_time}/shap_value',
        json.dumps(list(values),cls=Float32Encoder)
    )
    ex.log_scalar(
        f'divisive_shap_{dataset_name}_reapeat_{repeat_num}/{repeat_time}/time',
        t1 - t0
    )
    return t1 - t0


def onehot_to_location(y: np.array) -> np.array:
    res = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(len(y[i])):
            if y[i, j] > 0:
                res[i] = j
                break
    return res


@ex.automain
def main(train_size, valid_size, seed, datasets, repeat_time):
    t = 0.0
    for dataset_name in datasets:
        for i in range(repeat_time):
            fetcher = DataFetcher(dataset_name=dataset_name).split_dataset_by_count(
                train_size, valid_size
            )
            x_train, y_train, x_val, y_val, _, _ = fetcher.datapoints
            x_train = np.array(x_train.tolist())
            y_train = onehot_to_location(np.array(y_train.tolist()))
            x_val = np.array(x_val.tolist())
            y_val = onehot_to_location(np.array(y_val.tolist()))
            players = {"X":x_train,"y":y_train}
            test_set = {"X":x_val,"y":y_val}
            model = LogisticRegression()

            # model.fit(x_train, y_train)
            game = DataGame("classification", players, test_set, model)

            t += cal_sv(game, i + 1, repeat_time, dataset_name)   

        ex.log_scalar(
            f'divisive_shap_{dataset_name}/whole_time',
            t
        )