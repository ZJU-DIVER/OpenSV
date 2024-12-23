from typing import Optional, Any

import numpy as np
from tqdm import trange
from functools import partial
from multiprocessing import Pool

from scipy.optimize import minimize
from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray

def subtask(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        q: Array=None,
        sub_length: int=None
) -> tuple[Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = list(range(N))

    idxes = np.asarray(list(range(N)))
    b = np.zeros(N * sub_length).reshape((N, sub_length))
    u = np.zeros(sub_length)

    for t in trange(sub_length):
        length = rng.choice(q)
        rng.shuffle(idxes)
        for i in range(length):
            b[idxes[i]][t] = 1
        x_temp, y_temp = x_train[idxes[:length], :], y_train[idxes[:length]]
        u[t] = get_utility(x_temp, y_temp, x_valid, y_valid, clf)

    return (b, u)

def group_testing_ulimit_mp(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_proc: int=1,
        num_utility: int=500,
        truncated_threshold: float=0.00001,
        solver_iter: int=5000,
) -> Array:
    print(get_utility(x_train, y_train, x_valid, y_valid, clf), get_utility([], [], x_valid, y_valid, clf))
    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    
    T = num_utility - 1

    N = len(y_train)
    Z = 0
    for k in range(1, N):
        Z += 1.00 / k
    Z *= 2
    q = np.zeros(N - 1)
    for k in range(1, N):
        q[k - 1] = 1.00 / k + 1.00 / (N - k)
    q = q * (N / Z)
    q = q.round().astype(dtype=int).clip(min=1, max=N)
    
    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(subtask, x_train, y_train, x_valid, y_valid, clf, q)
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    
    b = np.concatenate([r[0] for r in ret], axis=1)
    u = np.concatenate([r[1] for r in ret], axis=0)

    delta_u = np.zeros(N * N).reshape((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            delta_u[i][j] = np.sum((b[i] - b[j]) * u) * Z / T
    
    print('Solving the feasibility problem')

    epsilon = truncated_threshold
    tolerance = epsilon / (2 * np.sqrt(N))
    target_u = get_utility(x_train, y_train, x_valid, y_valid, clf)

    def objective(s):
        return abs(target_u - np.sum(s))

    def constraint_func(s, i, j):
        return tolerance - abs((s[i] - s[j]) - delta_u[i][j])

    constraints = [{'type': 'ineq', 'fun': constraint_func, 'args': (i, j)} for i in range(N) for j in range(i + 1, N)]

    initial_guess = np.zeros(N) + (target_u / N)

    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': solver_iter}
    )
    print("Solver status:", result.success)
    print("Solver message:", result.message)
    val = result.x
    # val = val * (final_acc - init_acc) / np.sum(val)

    return val
