''' Data Cooperative Game '''
from copy import deepcopy,copy
from typing import Dict, Self

import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base import BaseGame
import numpy as np


class DataGame(BaseGame):
    players: Dict[str, np.ndarray]

    def __init__(self, task, players: Dict[str, np.ndarray], test_set: Dict[str, np.ndarray], model):
        super().__init__()
        self.task = task
        self.players = players
        self.model = model  # to create utility function
        self._size = len(self.players['y'])

        self.test_set = test_set

    @classmethod
    def create(cls, dataset='iris', task='classification', model=None, *args, **kwargs):
        # select model
        if kwargs['task'] == 'classification' and model is None:
            model = LogisticRegression(random_state=42, max_iter=100)
        elif kwargs['task'] == 'regression' and model is None:
            model = LinearRegression()
        else:
            raise ValueError(f'Unknown task: {kwargs["task"]}')

        # load data to players
        iris = datasets.load_iris()
        data, target = iris['data'], iris['target']
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
        players = {
            'X': X_train,
            'y': y_train,
        }
        test_set = {
            'X': X_test,
            'y': y_test,
        }
        return cls(task=task, players=players, test_set=test_set, model=model)

    @property
    def size(self) -> int:
        return self._size

    def get_utility(self, coalition):
        # coalition is the index of players
        temp_players = {
            'X': self.players['X'][coalition],
            'y': self.players['y'][coalition],
        }
        if len(np.unique(temp_players['y'])) == 1:
            return 1.0
        model = self.model
        model.fit(**temp_players)
        if self.task == 'classification':
            return accuracy_score(model.predict(self.test_set['X']), self.test_set['y'])
        else:
            raise ValueError(f'Unknown task: {self.task}')

    def copy(self) -> Self:
        return copy(self)

    def __copy__(self) -> Self:
        return deepcopy(self)

    def __repr__(self):
        return f'DataGame(task={self.task}, size={self.size})'
    
    def sv(self, sv):
        pass

    def utility_func(self, model):
        pass
