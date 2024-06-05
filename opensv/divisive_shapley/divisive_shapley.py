from typing import List, Callable, Tuple, Iterator
import numpy as np
from sklearn.cluster import KMeans
from functools import partial
from itertools import product, chain, combinations
from multiprocessing import Pool
from sklearn import neighbors
from tqdm import tqdm, trange

"""
References:
[1] Shapley Value Approximation with Divisive Clustering
[2] Shapley Values for dependent features using Divisive Clustering
"""


def power_set(iterable) -> Iterator:
    """Generate the power set of the all elements of an iterable obj.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(1, len(s) + 1))


def exact_shap(game, model):
    """
    Calculating the Shapley value of data points with exact method
    (Shapley value definition)

    :param model:    the selected model
    :return: Shapley value array `sv`
    :rtype: numpy.ndarray
    """

    n = game.num_players
    ext_sv = np.zeros(n)
    coef = np.zeros(n)
    fact = np.math.factorial
    coalition = np.arange(n)
    for s in range(n):
        coef[s] = fact(s) * fact(n - s - 1) / fact(n)
    sets = list(power_set(coalition))
    for idx in trange(len(sets)):
        u = game.get_utility(list(sets[idx]))
        for i in sets[idx]:
            ext_sv[i] += coef[len(sets[idx]) - 1] * u
        for i in set(coalition) - set(sets[idx]):
            ext_sv[i] -= coef[len(sets[idx])] * u
    return ext_sv


def shapley_value_exact(S: np.ndarray, v: Callable[[np.ndarray], float]) -> np.ndarray:
    # Placeholder for exact Shapley value calculation
    # Implement the actual Shapley value computation for coalition S
    # TODO: create a new game here and then invoke exact_shap

    return np.random.rand(len(S))


def characteristic_function(S: np.ndarray) -> float:
    # Placeholder for characteristic function
    return np.sum(S)  # Example: sum of values


class DivisiveShapley:
    def __init__(self, beta: float):
        self.beta = beta

    def divisive_shap_approx(self, S: np.ndarray, v: Callable[[np.ndarray], float]) -> np.ndarray:
        n = len(S)
        if n <= np.log(n) / np.log(self.beta):
            return shapley_value_exact(S, v)
        else:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(S)
            labels = kmeans.labels_
            unique_labels = np.unique(labels)
            if unique_labels.shape[0] == 1:
                return shapley_value_exact(S[labels == unique_labels[0]], v)
            shap_values = np.zeros(n)

            for label in unique_labels:
                subcoalition = S[labels == label]
                shap_values[labels == label] = self.divisive_shap_approx(subcoalition, v)

            # Synergy adjustments could be made here if necessary
            return shap_values

    def calculate(self, N: np.ndarray, v: Callable[[np.ndarray], float]) -> np.ndarray:
        return self.divisive_shap_approx(N, v)


if __name__ == "__main__":
    # Usage Example
    N = np.random.rand(100)  # Example data
    div_shap = DivisiveShapley(beta=2)
    shapley_values = div_shap.calculate(N, characteristic_function)
    print(shapley_values)
