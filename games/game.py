from abc import ABCMeta, abstractmethod

class Game(ABCMeta):
    @abstractmethod
    def size(self):
        return 0
    
    @abstractmethod
    def get_utility(self, coalition):
        return 0
