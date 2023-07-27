import sys
sys.path.append('..')
from sacred.observers import MongoObserver
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from typing import Tuple
import numpy as np
import json
from opensv import KNNShapley 
from opensv.config import ParamsTable 
from dataloader.preparedata_knn import *
from opensv.utils.utils import Float32Encoder
Array = np.ndarray

DATABASE_NAME = 'SV_Bench'
YOUR_CPU = "localhost:27017"

EXPERIMENT_NAME = 'my_experiment_knn-shapley'
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def dataset_config():
    dataset_name = '2dplanes'
    train_size = 1000
    valid_size = 100
    seed = 42


@ex.config
def running_config():
    num_procs = 1


@ex.capture
def cal_sv(x_train: Array,
           y_train: Array,
           x_val: Array,
           y_val: Array,
           k: int,
           solver: str,
           seed: int,
           num_procs: int,  # not used
           ):

    dv = KNNShapley()
    dv.load(x_train, y_train, x_val, y_val , ParamsTable(K=k))
    dv.solve(solver)
    ex.log_scalar(
        f'solver_{solver}_k-neighbors_{k}_',
        json.dumps(list(dv.get_values()),cls=Float32Encoder)
    )
    ex.log_scalar(
        f'solver_{solver}_k-neighbors_{k}_',
        str(dv.get_values())
    )


@ex.automain
def main(train_size, valid_size, seed,dataset_name):
    # step 1: data loader
    x_train, y_train, x_val, y_val =get_processed_data( dataset_name ,train_size,valid_size,seed)

    
    for k in [1 , 10, 16]:
        cal_sv(x_train, y_train, x_val, y_val, k , 'exact_sv')
