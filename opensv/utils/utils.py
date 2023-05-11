import numpy as np
from sklearn.metrics import accuracy_score
from typing import List


def get_utility(x_train, y_train, x_valid, y_valid, clf):
    """_summary_

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        x_valid (_type_): _description_
        y_valid (_type_): _description_
        clf (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        clf.fit(x_train, y_train)
        acc = accuracy_score(y_valid, clf.predict(x_valid))
    except ValueError:
        # Training set only has a single calss
        acc = accuracy_score(y_valid, [y_train[0]] * len(y_valid))
    return acc


def beta_weight(j: int, n: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    """gen weight for beta shapley

    Args:
        j (int): current pos !from 1 to n
        n (int): total num
        alpha (float): _description_
        beta (float): _description_


    Returns:
        float: beta weight
    """

    w = float(n)
    for k in range(1, n):  # from 1 to n - 1
        w /= alpha + beta + k - 1
        if k <= j - 1:
            w *= beta + k - 1
            # solve C(n-1 j-1)-1 by multiply C(n-1 j-1)
            w *= (n - k) / k
        else:
            # from j to n - 1
            temp_k = k - j + 1
            w *= alpha + temp_k - 1
    return w


def beta_weight_list(n: int, alpha: float = 1.0, beta: float = 1.0) -> List[float]:
    w_list = []
    for j in range(1, n + 1):
        w_list.append(beta_weight(alpha, beta, j, n))
    return w_list


def check_convergence(mem, n_chains: int = 10):
    """
    From https://github.com/ykwon0407/beta_shapley/blob/master/betashap/utils.py
    Compute Gelman-Rubin statistic
    Ref. https://arxiv.org/pdf/1812.09384.pdf (p.7, Eq.4)
    """
    if len(mem) < 1000:
        return 100
    n_chains = 10
    (N, n_to_be_valued) = mem.shape
    if (N % n_chains) == 0:
        n_MC_sample = N // n_chains
        offset = 0
    else:
        n_MC_sample = N // n_chains
        offset = (N % n_chains)
    mem = mem[offset:]
    percent = 25
    while True:
        IQR_contstant = np.percentile(mem.reshape(-1), 50 + percent) - np.percentile(mem.reshape(-1), 50 - percent)
        if IQR_contstant == 0:
            percent += 10
            if percent == 105:
                assert False, 'CHECK!!! IQR is zero!!!'
        else:
            break

    mem_tmp = mem.reshape(n_chains, n_MC_sample, n_to_be_valued)
    GR_list = []
    for j in range(n_to_be_valued):
        mem_tmp_j_original = mem_tmp[:, :, j].T  # now we have (n_MC_sample, n_chains)
        mem_tmp_j = mem_tmp_j_original / IQR_contstant
        mem_tmp_j_mean = np.mean(mem_tmp_j, axis=0)
        s_term = np.sum((mem_tmp_j - mem_tmp_j_mean) ** 2) / (
                    n_chains * (n_MC_sample - 1))  # + 1e-16 this could lead to wrong estimator

        mu_hat_j = np.mean(mem_tmp_j)
        B_term = n_MC_sample * np.sum((mem_tmp_j_mean - mu_hat_j) ** 2) / (n_chains - 1)

        GR_stat = np.sqrt((n_MC_sample - 1) / n_MC_sample + B_term / (s_term * n_MC_sample))
        GR_list.append(GR_stat)
    GR_stat = np.max(GR_list)
    print(f'Total number of random sets: {len(mem)}, GR_stat: {GR_stat}', flush=True)
    return GR_stat
