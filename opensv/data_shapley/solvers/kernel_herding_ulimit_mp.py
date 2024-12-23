from typing import Optional, Any

import numpy as np
from tqdm import tqdm, trange
from functools import partial
from multiprocessing import Pool

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def p_expectation_mallows(d, lambda_param):
    numerator = 1
    denominator = 1 - np.exp(-lambda_param / (d * (d - 1) / 2))

    for j in range(1, d + 1):
        numerator *= 1 - np.exp(-lambda_param * j / (d * (d - 1) / 2))
        denominator *= j

    return numerator / denominator


def p_expectation_kendall():
    return 0


def p_expectation_spearman(d):
    return d * (d + 1)**2 / 4


def compute_p_expectation(kernel_type, d, lambda_param=None):
    if kernel_type == "mallows":
        if lambda_param is None:
            raise ValueError("lambda_param is required for Mallows kernel.")
        return p_expectation_mallows(d, lambda_param)
    elif kernel_type == "kendall":
        return p_expectation_kendall()
    elif kernel_type == "spearman":
        return p_expectation_spearman(d)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def kernel_herding(kernel_func, p_expectation, candidate_num, d):
    samples = []
    scores = []

    candidate_points = [np.random.permutation(d) for _ in range(candidate_num)]  # Generate candidate_num candidates randomly

    for n in trange(candidate_num):
        max_score = -np.inf
        best_candidate = None

        for candidate in candidate_points:
            score = p_expectation - (1 / (n + 1)) * sum(kernel_func(candidate, s) for s in samples)

            if score > max_score:
                max_score = score
                best_candidate = candidate

        samples.append(best_candidate)
        scores.append(max_score)

    return samples, scores


def mallows_kernel(x, y, lambda_param=1):
    n_discordant_pairs = sum((x[i] > x[j]) != (y[i] > y[j]) for i in range(len(x)) for j in range(i + 1, len(x)))
    return np.exp(-lambda_param * n_discordant_pairs / len(x)**2)


def kendall_kernel(x, y):
    n_concordant_pairs = sum((x[i] > x[j]) == (y[i] > y[j]) for i in range(len(x)) for j in range(i + 1, len(x)))
    n_discordant_pairs = sum((x[i] > x[j]) != (y[i] > y[j]) for i in range(len(x)) for j in range(i + 1, len(x)))
    return (n_concordant_pairs - n_discordant_pairs) / (len(x) * (len(x) - 1) / 2)


def spearman_kernel(x, y):
    rank_diff = np.array(x) - np.array(y)
    return -0.5 * np.sum(rank_diff**2)

def subtask(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        init_acc: float=0,
        final_acc: float=1,
        perm_list: Array=None,
        sub_range: range=None,
) -> Array:
    N = len(y_train)
    val = np.zeros(N)
    for k in tqdm(sub_range):
        idxes = perm_list[k]
        acc = init_acc
        for i in range(1, N + 1):
            x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
            new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
            val[idxes[i-1]] += new_acc - acc
            acc = new_acc
    return val

def kernel_herding_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
        kernel_type: str="mallows",
        lambda_param: int=0
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    
    N = len(y_train)
    T = (num_utility - 2) // N

    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    print(final_acc - init_acc)

    if lambda_param == 0:
        lambda_param = N

    if kernel_type == "mallows":
        p_expectation = compute_p_expectation(kernel_type, N, lambda_param=lambda_param)
        kernel_func = lambda x, y: mallows_kernel(x, y, lambda_param)
    elif kernel_type == "kendall":
        p_expectation = compute_p_expectation(kernel_type, N)
        kernel_func = kendall_kernel
    elif kernel_type == "spearman":
        p_expectation = compute_p_expectation(kernel_type, N)
        kernel_func = spearman_kernel
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    samples, scores = kernel_herding(kernel_func, p_expectation, T, N)

    sub_length = split_permutation_num(T, num_proc)
    cur = 0
    sub_range = []
    for i in sub_length:
        sub_range.append(range(cur, cur + i))
        cur += i

    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, init_acc, final_acc, samples)
    ret = pool.map(func, sub_range)
    pool.close()
    pool.join()

    ret_val = np.asarray(ret)
    val = ret_val.sum(axis=0) / T
    # val = val * (final_acc - init_acc) / np.sum(val)

    return val