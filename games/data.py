"""
opensv.games.data
~~~~~~~~~~~~~~~~~

This module provides various games for data valuation tasks.
"""
from typing import Any

from .base import BaseGame


class DataValuation(BaseGame):
    def __repr__(self) -> None:
        pass

    def __init__(self, train_data, valid_data, model):
        self.train_data = train_data
        self.valid_data = valid_data
        self.model = model

        self.n = len(self.train_data)
        
    def size(self):
        return self.n
    
    def get_utility(self, coalition):
        return super().get_utility(coalition)
    
    @classmethod
    def create_game(cls, data, valid_data=None, task=None, model=None):
        if task == 'classification':
            pass
        else:
            pass
        
        return DataGame(
            data,
            valid_data,
            model
        )


class GroupDataValuation(BaseGame):
    def __repr__(self) -> None:
        pass

    def get_utility(self, coalition) -> Any:
        pass

    @property
    def size(self) -> int:
        pass





