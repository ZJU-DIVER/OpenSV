from typing import Optional, Any

import numpy as np
from tqdm import trange
from random import randint, sample
from opensv.utils.utils import get_utility_classwise, class_conditional_sampling

Array = np.ndarray

def truncated_mc(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        truncated_threshold: float=0.001
) -> Array:
    T = num_perm
    epsilon = truncated_threshold
    N = len(y_train)
    val = np.zeros(N)
    labels = list(set(y_train))
    for label in labels:
        # Select data based on the class label
        orig_indices = np.array(list(range(x_train.shape[0])))[y_train == label]
        x_train_label = x_train[y_train == label]
        y_train_label = y_train[y_train == label]
        x_train_nonlabel = x_train[y_train != label]
        y_train_nonlabel = y_train[y_train != label]
        nonlabel_set = list(set(y_train_nonlabel))
        for _ in trange(num_perm):
            # Sample a subset of training data from other labels for each i
            if len(nonlabel_set) == 1:
                s = randint(1, x_train_nonlabel.shape[0])
                Idx_nonlabel = sample(list(range(x_train_nonlabel.shape[0])), s)
            else:
                Idx_nonlabel = class_conditional_sampling(y_train_nonlabel, list(set(y_train_nonlabel)))
            trnX_nonlabel_i = x_train_nonlabel[Idx_nonlabel, :]
            trnY_nonlabel_i = y_train_nonlabel[Idx_nonlabel]
            # With all data from the target class and the sampled data from other classes
            x_temp = np.concatenate((trnX_nonlabel_i, x_train_label))
            y_temp = np.concatenate((trnY_nonlabel_i, y_train_label))
            clf.fit(x_temp, y_temp)
            final_acc = get_utility_classwise(x_temp, y_temp, x_valid, y_valid, label, clf)

            acc = 0  # init value
            idxes = list(range(N))
            np.random.shuffle(idxes)
            for i in range(1, N + 1):
                if abs(final_acc - acc) < epsilon:
                    break
                try:
                    x_temp = np.concatenate((trnX_nonlabel_i, x_train[idxes[:i], :]))
                    y_temp = np.concatenate((trnY_nonlabel_i, y_train[idxes[:i]]))
                    new_acc = get_utility_classwise(x_temp, y_temp, x_valid, y_valid, label, clf)
                    if idxes[i - 1] in orig_indices:
                        val[idxes[i - 1]] += new_acc - acc
                    acc = new_acc
                except:
                    pass
    return val / T
