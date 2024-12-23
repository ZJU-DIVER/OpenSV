from typing import Optional, Any

import numpy as np
from tqdm import trange

from scipy.optimize import minimize
from opensv.utils.utils import get_utility

Array = np.ndarray

def group_testing(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        truncated_threshold: float=0.00001,
) -> Array:
    T = num_perm

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
    idxes = np.asarray(list(range(N)))

    b = np.zeros(N * T).reshape((N, T))
    u = np.zeros(T)
    for t in trange(T):
        length = np.random.choice(q)
        np.random.shuffle(idxes)
        for i in range(length):
            b[idxes[i]][t] = 1
        x_temp, y_temp = x_train[idxes[:length], :], y_train[idxes[:length]]
        u[t] = get_utility(x_temp, y_temp, x_valid, y_valid, clf)

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
        options={'maxiter': 5000}
    )
    print("Solver status:", result.success)
    print("Solver message:", result.message)
    val = result.x

    return val
