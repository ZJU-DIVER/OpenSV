# impl refer to official repo of [Beta-Shapley]
# (https://github.com/ykwon0407/beta_shapley/blob/master/betashap/ShapEngine.py)

from typing import Optional, Dict

import numpy as np
from tqdm import trange

from ..utils.utils import get_utility, beta_weight_list, check_convergence


def beta_shapley(x_train, y_train, x_valid, y_valid, clf, params: Optional[Dict] = None):
    """_summary_

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        x_valid (_type_): _description_
        y_valid (_type_): _description_
        clf (_type_): _description_
        params (Optional[Dict], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Extract params
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    rho = params.get('rho', 1.0005)  # terminating threshold
    T = params.get('beta_iter', 10)

    N = len(y_train)
    weight_list_default = np.array(beta_weight_list(N)) / N  # normalizing
    weight_list = np.array(beta_weight_list(N, alpha, beta)) / N
    idxes = list(range(N))
    hat_rho = 2 * rho

    # For convergence check
    prob_marginal_contrib = np.zeros((N, N))
    prob_marginal_contrib_count = np.zeros((N, N))

    B = 1
    marginal_contribs = np.zeros((0, N))
    for _ in trange(T):  # T for a forced max iter
        np.shuffle(idxes)
        acc = 0
        marginal_contribs_tmp = np.zeros(N)
        for k, j in enumerate(idxes):
            temp_idxes = idxes[:k + 1]
            acc_new = get_utility(x_train[temp_idxes, :], y_train[temp_idxes],
                                  x_valid, y_valid, clf)

            marginal_contribs_tmp[j] = (acc_new - acc) / weight_list_default[k]
            prob_marginal_contrib[j, k] += acc_new - acc
            prob_marginal_contrib_count[j, k] += 1
            acc = acc_new

        marginal_contribs = np.concatenate([marginal_contribs, marginal_contribs_tmp.reshape(1, -1)])
        # Update the Gelman-Rubin statistic rho_hat (every 100 permutations)
        if B % 100 == 0:
            hat_rho = check_convergence(marginal_contribs)
            if hat_rho < rho:
                # terminate the outer loop earlier
                break

    # Calculate beta Shapley with marginal contribution
    shap_value_weight = np.sum(prob_marginal_contrib * weight_list, axis=1)
    val = shap_value_weight / np.mean(np.sum(prob_marginal_contrib_count, axis=1))
    return val
