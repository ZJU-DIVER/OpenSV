from typing import Optional, Any

import numpy as np
from tqdm import tqdm, trange
from functools import partial
from multiprocessing import Pool
from scipy.stats import qmc

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def sobol_to_sphere(points):
    n_samples, d_minus_2 = points.shape
    d_minus_1 = d_minus_2 + 1
    spherical_points = np.zeros((n_samples, d_minus_1))

    for i, point in enumerate(points):
        angles = np.zeros(d_minus_2 + 1)
        for j in range(d_minus_2):
            angles[j] = np.arccos(1 - 2 * point[j])
        angles[-1] = 2 * np.pi * point[-1]

        cartesian = np.zeros(d_minus_1)
        cartesian[0] = np.cos(angles[0])
        for k in range(1, d_minus_1 - 1):
            cartesian[k] = np.prod(np.sin(angles[:k])) * np.cos(angles[k])
        cartesian[-1] = np.prod(np.sin(angles))

        spherical_points[i] = cartesian

    return spherical_points


def create_projection_matrix(d):
    U = np.zeros((d, d - 1))
    for i in range(d):
        for j in range(d - 1):
            if i == j:
                U[i, j] = d - 1
            elif i < j:
                U[i, j] = -1
    return U / np.linalg.norm(U, axis=0)


def spherical_to_permutation(vector, U):
    projected = U @ vector
    permutation = np.argsort(-projected)
    return permutation


def sobol_permutations(n_samples, d):
    sobol_engine = qmc.Sobol(d=d - 2, scramble=True)
    sobol_points = sobol_engine.random(n=n_samples)
    spherical_points = sobol_to_sphere(sobol_points)
    
    assert spherical_points.shape[1] == d - 1, "Spherical points must have dimension d-1."

    U = create_projection_matrix(d)
    permutations = [spherical_to_permutation(point, U) for point in spherical_points]
    return permutations



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


def sobol_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    
    N = len(y_train)
    T = (num_utility - 2) // N

    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    print(final_acc - init_acc)

    samples = sobol_permutations(T, N)

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