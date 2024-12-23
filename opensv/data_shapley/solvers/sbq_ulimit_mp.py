from typing import Optional, Any

import numpy as np
from tqdm import tqdm, trange
from functools import partial
from multiprocessing import Pool

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def sequential_bayesian_quadrature(kernel_func, candidate_num, d, sampling_dist, integrand):
    samples = []
    kernel_matrix_inv = None
    expectations = []

    x0 = sampling_dist(d)
    samples.append(x0)

    expectation = np.mean([kernel_func(x0, x) for x in sampling_dist(d, size=100)])
    expectations.append(expectation)

    kernel_matrix_inv = np.linalg.inv(np.array([[kernel_func(x0, x0)]]))

    for _ in trange(1, candidate_num):
        best_sample = None
        min_variance = float('inf')

        for _ in range(100):
            candidate = sampling_dist(d)
            k = np.array([kernel_func(candidate, sample) for sample in samples])
            variance = kernel_func(candidate, candidate) - np.dot(k.T, np.dot(kernel_matrix_inv, k))

            if variance < min_variance:
                min_variance = variance
                best_sample = candidate

        if any((best_sample == sample).all() for sample in samples):
            continue

        samples.append(best_sample)
        k = np.array([kernel_func(best_sample, sample) for sample in samples[:-1]])
        expectations.append(np.mean([kernel_func(best_sample, x) for x in sampling_dist(d, size=100)]))

        K = np.zeros((len(samples), len(samples)))
        for i in range(len(samples)):
            for j in range(len(samples)):
                K[i, j] = kernel_func(samples[i], samples[j])
        K += 1e-8 * np.eye(len(samples))
        kernel_matrix_inv = np.linalg.inv(K)

    weights = np.dot(kernel_matrix_inv, np.array(expectations))

    return weights, samples


def mallows_kernel(x, y, lambda_param=1):
    n_discordant_pairs = sum((x[i] > x[j]) != (y[i] > y[j]) for i in range(len(x)) for j in range(i + 1, len(x)))
    return np.exp(-lambda_param * n_discordant_pairs / len(x)**2)

def uniform_sampling(d, size=1):
    if size == 1:
        return np.random.permutation(d)
    else:
        return [np.random.permutation(d) for _ in range(size)]

def integrand_function(x):
    return sum(x)


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


def sbq_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
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

    samples = []
    kernel_func = lambda x, y: mallows_kernel(x, y, lambda_param)

    # Run Sequential Bayesian Quadrature
    weights, samples = sequential_bayesian_quadrature(
        kernel_func=kernel_func,
        candidate_num=T,
        d=N,
        sampling_dist=uniform_sampling,
        integrand=integrand_function,
    )

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