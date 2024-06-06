from typing import List, Callable, Tuple, Iterator
import numpy as np
from sklearn.cluster import KMeans
from functools import partial
from itertools import product, chain, combinations
from multiprocessing import Pool
from sklearn import neighbors
from tqdm import tqdm, trange
import math
from opensv.games import *
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


def exact_shap(game: BaseGame):
    n = game.size
    ext_sv = np.zeros(n)
    coef = np.zeros(n)
    fact = math.factorial
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


def shapley_value_exact(game: BaseGame) -> np.ndarray:
    return exact_shap(game)


class DivisiveShapley:
    def __init__(self, beta: float):
        self.beta = beta  # the only hyperparameter of divisive Shapley

    def divisive_shap_approx(self, game: BaseGame) -> np.ndarray:
        n = game.size
        if n <= np.log(n) / np.log(self.beta) or n == 1:
            return shapley_value_exact(game)
        else:
            if isinstance(game, DataGame):
                kmeans = KMeans(n_clusters=2, random_state=0).fit(game.players["X"], game.players["y"])
                labels = kmeans.labels_
                unique_labels = np.unique(labels)
                if unique_labels.shape[0] == 1:
                    return shapley_value_exact(game)
                shap_values = np.zeros_like(game.players["y"])

                for label in unique_labels:
                    index = (labels == label)
                    sub_coalition = {
                        "X": game.players["X"][index],
                        "y": game.players["y"][index],
                    }
                    sub_game = game.__copy__()
                    sub_game.players = sub_coalition
                    sub_game._size = len(sub_game.players['y'])
                    shap_values[labels == label] = self.divisive_shap_approx(sub_game)
                # after we set shapley value, do normalization
                u = game.get_utility(np.arange(game.size))
                shap_values *= u / shap_values.sum()
            else:
                raise NotImplementedError("have not supported game type")
            # Synergy adjustments could be made here if necessary
            return shap_values

    def calculate(self, game: BaseGame) -> np.ndarray:
        return self.divisive_shap_approx(game)


if __name__ == "__main__":
    # Usage Example
    pass
    # N = np.random.rand(100)  # Example data
    # div_shap = DivisiveShapley(beta=2)
    # shapley_values = div_shap.calculate(game)
    # print(shapley_values)
