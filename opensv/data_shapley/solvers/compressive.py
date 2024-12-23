from typing import Optional, Any

import numpy as np
from tqdm import trange

from scipy.optimize import minimize
from opensv.utils.utils import get_utility

Array = np.ndarray

def compressive(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int=500,
        num_measure: int=500,
        truncated_threshold: float=0.0001
) -> Array:
    T = num_perm
    N = len(y_train)
    M = num_measure

    A = np.random.randint(2, size=(M, N)).astype(float)
    A = (2 * A - 1) / np.sqrt(M)
    idxes = np.asarray(list(range(N)))
    phi = np.zeros(T * N).reshape((T, N))
    y = np.zeros(M * T).reshape((M, T))

    for t in trange(T):
        np.random.shuffle(idxes)
        acc = 0
        for i in range(1, N + 1):
            x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
            new_acc = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
            phi[t][idxes[i - 1]] = new_acc - acc
            acc = new_acc
        for m in range(M):
            y[m][t] = np.sum(A[m] * phi[t])
    
    yp = np.zeros(M)
    for m in range(M):
        yp[m] = sum(y[m]) / T
        
    print('Solving the feasibility problem')
    
    sp = get_utility(x_train, y_train, x_valid, y_valid, clf) / N
    epsilon = truncated_threshold
    
    def objective(s):
        return np.sum(np.abs(s))

    def constraint_func(s):
        vec = A @ (s + sp) - yp
        return epsilon - np.sqrt(vec.dot(vec))

    constraints = [{'type': 'ineq', 'fun': constraint_func}]

    initial_guess = np.zeros(N)

    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 5000}
    )
    print("Solver status:", result.success)
    print("Solver message:", result.message)

    val = result.x + sp

    return val
