import sys
sys.path.append("..")

import torch
import torch.nn.functional as F
from opendataval.dataloader import DataFetcher
from sklearn import preprocessing
import numpy as np
Array = np.ndarray

def onehot_to_location(y: Array) -> Array:
    res = np.zeros((y.shape[0]), dtype=int)
    for i in range(y.shape[0]):
        for j in range(len(y[i])):
            if y[i, j] > 0:
                res[i] = j
                break
    return res


fetcher = DataFetcher(dataset_name='2dplanes', random_state=42).split_dataset_by_count(
    50, 20
)
x_train, y_train, x_val, y_val, _, _ = fetcher.datapoints
x_train = np.array(x_train.tolist())
y_train = onehot_to_location(np.array(y_train.tolist()))
x_val = np.array(x_val.tolist())
y_val = onehot_to_location(np.array(y_val.tolist()))

from opensv import DataShapley
from opensv.config import ParamsTable

shap = DataShapley()
shap.load(x_train, y_train, x_val, y_val, para_tbl=ParamsTable(num_proc=48, num_perm=10000))
shap.solve('svarm_stratified_mp')
base_value = np.array(shap.get_values())
base_value = base_value.reshape(-1, 1)
scaler = preprocessing.StandardScaler().fit(base_value)
base_value = scaler.transform(base_value)
base_value = torch.tensor(base_value)

shap2 = DataShapley()
shap2.load(x_train, y_train, x_val, y_val, para_tbl=ParamsTable(num_proc=48, num_perm=10000))
shap2.solve('monte_carlo_mp')
val_value = np.array(shap2.get_values())
val_value = val_value.reshape(-1, 1)
scaler = preprocessing.StandardScaler().fit(val_value)
val_value = scaler.transform(val_value)
val_value = torch.tensor(val_value)

print(base_value)
print(val_value)
mse = F.mse_loss(base_value, val_value)
print(f"mse: {mse:.8f}")
