'''
Note！！！
'preparedate.py' is for experimental purpose only and dataloader will  be  completed  in the future
'''

import sys
import numpy as np
import pandas as pd
import pickle

OpenML_dataset = ['fraud', 'apsfail', 'click', 'phoneme', 'wind', 'pol', 'creditcard', 'cpu', 'vehicle', '2dplanes']
big_dataset = ['MNIST', 'CIFAR10', 'Dog_vs_Cat', 'Dog_vs_CatFeature', 'FMNIST']

def normalize(val):
  v_max, v_min = np.max(val), np.min(val)
  val = (val-v_min) / (v_max - v_min)
  return val

def get_processed_data(dataset, n_data ,  n_val , seed):
    
    print('-------')
    print('Load Dataset {}'.format(dataset))

    if dataset in OpenML_dataset:
        X, y, _, _ = get_data(dataset , seed)
        x_train, y_train = X[:n_data], y[:n_data]
        x_val, y_val = X[n_data:n_data+n_val], y[n_data:n_data+n_val]

        X_mean, X_std= np.mean(x_train, 0), np.std(x_train, 0)
        normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
        x_train, x_val = normalizer_fn(x_train), normalizer_fn(x_val)

    else:
        x_train, y_train, x_test, y_test = get_data(dataset , seed)
        x_val, y_val = x_test, y_test
        '''
        if dataset != 'covertype':
            x_train, y_train = make_balance_sample_multiclass(x_train, y_train, n_data)
            x_val, y_val = make_balance_sample_multiclass(x_test, y_test, n_data)
        '''
    return x_train, y_train, x_val, y_val


def get_data(dataset , seed):

    if dataset in ['covertype']+OpenML_dataset:
        x_train, y_train, x_test, y_test = get_minidata(dataset , seed)
    '''
    elif dataset == 'MNIST':
        x_train, y_train, x_test, y_test = get_mnist()
    elif dataset == 'CIFAR10':
        x_train, y_train, x_test, y_test = get_dogcatFeature()
    elif dataset == 'FMNIST':
        x_train, y_train, x_test, y_test = get_fmnist()
    else:
        sys.exit(1)
    '''
    return x_train, y_train, x_test, y_test


def get_minidata(dataset , seed):

    open_ml_path = '..datasets/'

    np.random.seed(seed)

    if dataset == 'covertype':
        x_train, y_train, x_test, y_test = pickle.load( open('covertype_200.dataset', 'rb') )

    elif dataset == 'fraud':
        data_dict=pickle.load(open(open_ml_path+'CreditCardFraudDetection_42397.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'apsfail':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'APSFailure_41138.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'click':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'Click_prediction_small_1218.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'phoneme':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'phoneme_1489.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'wind':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'wind_847.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'pol':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'pol_722.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'creditcard':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'default-of-credit-card-clients_42477.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'cpu':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'cpu_act_761.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == 'vehicle':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'vehicle_sensIT_357.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    elif dataset == '2dplanes':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'2dplanes_727.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        

    else:
        print('No such dataset!')
        sys.exit(1)


    if dataset not in ['covertype']:
        idxs=np.random.permutation(len(data))
        data, target=data[idxs], target[idxs]
        return data, target, None, None
    else:
        return x_train, y_train, x_test, y_test



