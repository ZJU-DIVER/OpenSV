from abc import ABCMeta, abstractmethod


class BaseGame(ABCMeta):
    @property
    def type(self):
        return NotImplemented

    @property
    @abstractmethod
    def size(self):
        raise NotImplemented

    @property
    def sv(self):
        raise NotImplemented

    @sv.setter
    @abstractmethod
    def sv(self, sv):
        raise NotImplemented

    @property
    def players(self):
        """
        n players for a game, each player can be
            - a data tuple / a group of data tuples / an ML model / a prompt for valuation
            - a plane / a voter / a shoe for traditional game
            - a feature / a feature value in a data tuple for feature attribution
        """
        return NotImplemented

    @players.setter
    @abstractmethod
    def players(self, players):
        pass

    @property
    def utility_func(self):
        """
        utility function is a function get the coalition index and return utility
        """
        return NotImplemented

    @utility_func.setter
    @abstractmethod
    def utility_func(self, model):
        pass
    
    @abstractmethod
    def get_utility(self, coalition):
        raise NotImplemented

    @abstractmethod
    def __repr__(self):
        raise NotImplemented
