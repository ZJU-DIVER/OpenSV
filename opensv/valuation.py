from sklearn.linear_model import LogisticRegression as LR

from .config import Config

from .values.loo import loo
from .values.tmc_shapley import tmc_shapley
from .values.beta_shapley import beta_shapley


class Valuation(object):
    def __init__(self, x_train, y_train, x_valid=None, y_valid=None):
        """_summary_

        Args:
            x_train (_type_): _description_
            y_train (_type_): _description_
            x_valid (_type_, optional): _description_. Defaults to None.
            y_valid (_type_, optional): _description_. Defaults to None.
        """
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid

        if x_valid is None:
            self.x_valid, self.y_valid = x_train, y_train

        self.values = {}  # valuation results
        self.clf = None  # instance of classifier

        config = Config()
        self.params = config.get_params()

    def get_values(self):
        if self.values is not None:
            return self.values
        else:
            raise ValueError("[!] No values computed")

    def estimate(self, clf=None, method='loo', params=None):
        """_summary_

        Args:
            clf (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to 'loo'.
            params (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.values = {}

        self.clf = clf
        if clf is None:
            self.clf = LR(solver="liblinear", max_iter=500, random_state=0)

        if params is not None:
            print("[+] Overload parameters")
            print(f"{'key':<15} {'Value':<15}")
            for k, v in params.items():
                print(f"{k:<15} {v:<15}")
            self.params = params

        # Call different data valuation approaches
        args = [self.x_train, self.y_train, self.x_valid, self.y_valid, self.clf, self.params]
        match method:
            case "loo":
                vals = loo(*args)
            case "tmc-shapley":
                vals = tmc_shapley(*args)
            case "beta-shapley":
                vals = beta_shapley(*args)
            case _:
                raise ValueError(f"[!] Unrecognized data valuation method: {method}")

        self.values = vals
