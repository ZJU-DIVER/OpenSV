''' Data Cooperative Game '''

from .game import Game 

class DataGame(Game):
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