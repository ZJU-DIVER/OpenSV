"""
opensv.games.base
~~~~~~~~~~~~~~~~~

This module provide base abstract class for different cooperative games.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Callable

import numpy as np


class BaseGame(ABC):
    @property
    def type(self) -> str:
        return NotImplemented

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplemented

    @property
    def sv(self) -> List[float] | np.ndarray:
        raise NotImplemented

    @sv.setter
    @abstractmethod
    def sv(self, sv) -> None:
        return self._sv

    @property
    def players(self) -> List[Any]:
        """
        n players for a game, each player can be
            - a data tuple / a group of data tuples / an ML model / a prompt for valuation
            - a plane / a voter / a shoe for traditional game
            - a feature / a feature value in a data tuple for feature attribution
        """
        return self._players

    @players.setter
    @abstractmethod
    def players(self, players) -> None:
        pass

    @property
    def utility_func(self) -> Callable[[Any], Any]:
        """
        utility function is a function get the coalition index and return utility
        """
        return self._u

    @utility_func.setter
    @abstractmethod
    def utility_func(self, model) -> None:
        pass
    
    @abstractmethod
    def get_utility(self, coalition) -> Any:
        raise NotImplemented

    @abstractmethod
    def __repr__(self) -> None:
        raise NotImplemented
