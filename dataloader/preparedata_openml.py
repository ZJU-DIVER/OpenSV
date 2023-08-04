'''
Note！！！
'preparedate.py' is for experimental purpose only and dataloader will  be  completed  in the future
'''

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

base='../datasets'

def get_processed_data(dataset, id, n_data ,  n_val , seed):
    print('-------')
    print('Load Dataset {}'.format(dataset))
    np.random.seed(seed)
    data, target = load_openml(id)
    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs]

    x_train, y_train = data[:n_data], target[:n_data]
    x_val, y_val = data[n_data:n_data+n_val], target[n_data:n_data+n_val]
    # print(x_train, y_train)
    return x_train, y_train, x_val, y_val 


def load_openml(id, is_classification=True, file_home = base):
    """
        come from opendataval 0.0 hahahaha
    """
    dataset = fetch_openml(data_id=id, as_frame=False,data_home=file_home)
    category_list = list(dataset["categories"].keys())
    if len(category_list) > 0:
        category_indices = [dataset["feature_names"].index(x) for x in category_list]
        noncategory_indices = [
            i for i in range(len(dataset["feature_names"])) if i not in category_indices
        ]
        X, y = dataset["data"][:, noncategory_indices], dataset["target"]
    else:
        X, y = dataset["data"], dataset["target"]

    # label transformation
    if is_classification is True:
        list_of_classes, y = np.unique(y, return_inverse=True)
        print(list_of_classes)
        print(y)
    else:
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)  # standardization
    return X, y


