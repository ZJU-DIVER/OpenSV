from typing import Optional, Any

import numpy as np
from tqdm import tqdm, trange
from functools import partial
from multiprocessing import Pool

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b) * b for b in basis)
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def generate_spherical_codes(d):
    random_matrix = np.random.randn(d - 1, d - 1)
    orthogonal_basis = gram_schmidt(random_matrix)
    spherical_samples = np.vstack([orthogonal_basis, -orthogonal_basis])
    return spherical_samples


def spherical_to_permutation(vector, U):
    projected = U @ vector
    permutation = np.argsort(-projected)
    return permutation


def sample_permutations_from_sphere(d):
    spherical_codes = generate_spherical_codes(d)

    U = np.zeros((d, d-1))
    for i in range(d):
        for j in range(d-1):
            if i == j:
                U[i, j] = d - 1
            elif i < j:
                U[i, j] = -1
    U = U / np.linalg.norm(U, axis=0)
    
    permutations = [spherical_to_permutation(code, U) for code in spherical_codes]
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

def orthogonal_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    
    N = len(y_train)
    T = (num_utility - 2) // N

    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    print(final_acc - init_acc)
    
    samples = []
    while len(samples) < T:
        samples = samples + sample_permutations_from_sphere(N)

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