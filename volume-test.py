import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import opensv



def load_credit_card(n_participants=3, s=30, train_test_diff_distr=False):
    """     
	    Method to load the credit_card dataset.
        Args:
            n_participants (int): number of data subsets to generate
            s (int): number of data samples for each participant (equal)
            train_test_diff_distr (bool): whether to generate a test set that has a different distribution from the train set
            path_prefix (str): prefix for the file path
        Returns:
            feature_datasets, labels, feature_datasets_test, test_labels: each a list containing the loaded dataset
    """
    test_s = int(s * 0.2)
    
    data = pd.read_csv('datasets/creditcard.csv')
    
    # Use high amounts as hold out set
    hold_out_idx = list(data[data['Amount'] > 1000].index)
    hold_out = data.iloc[hold_out_idx[:test_s*n_participants]]
    data = data.drop(hold_out_idx)
    
    # Drop redundant features
    data = data.drop(['Class', 'Time'], axis = 1)
    hold_out = hold_out.drop(['Class', 'Time'], axis = 1)
    
    data = shuffle(data)
    X = data.iloc[:, data.columns != 'Amount']
    y = data.iloc[:, data.columns == 'Amount']

    # Feature selection
    cols = ['V1', 'V2', 'V5', 'V7', 'V10', 'V20', 'V21', 'V23']
    
    X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.2, random_state=1234)
    if train_test_diff_distr:
        X_test = hold_out.iloc[:, hold_out.columns != 'Amount'][cols]
        y_test = hold_out.iloc[:, hold_out.columns == 'Amount']
    
    feature_datasets, feature_datasets_test, labels, test_labels = [], [], [], []
    
    for i in range(n_participants):
        start_idx = i * s
        end_idx = (i + 1) * s
        test_start_idx = i * test_s
        test_end_idx = (i + 1) * test_s
        feature_datasets.append(torch.from_numpy(np.array(X_train[start_idx:end_idx], dtype=np.float32)))
        labels.append(torch.from_numpy(np.array(y_train[start_idx:end_idx], dtype=np.float32).reshape(-1, 1)))
        feature_datasets_test.append(torch.from_numpy(np.array(X_test[test_start_idx:test_end_idx], dtype=np.float32)))
        test_labels.append(torch.from_numpy(np.array(y_test[test_start_idx:test_end_idx], dtype=np.float32).reshape(-1, 1)))
    
    return feature_datasets, labels, feature_datasets_test, test_labels

n_participants = M = 100
D = 8
train_sizes = [200, 200, 200]
test_sizes = [200] * M
train_test_diff_distr=False

ranges=None
#feature_datasets = get_synthetic_datasets(n_participants=M, sizes=train_sizes, d=D, ranges=ranges)
#feature_datasets_test = get_synthetic_datasets(n_participants=M, sizes=test_sizes, d=D, ranges=ranges)


assert D == 8

feature_datasets, labels, feature_datasets_test, test_labels = load_credit_card(n_participants=M, s=50, train_test_diff_distr=train_test_diff_distr)

feature_datasets=torch.tensor([item.numpy() for item in feature_datasets])
shap = opensv.RVShapley()
shap.load(feature_datasets)
shap.solve()
print(shap.get_values())