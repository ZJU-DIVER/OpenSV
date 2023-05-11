from typing import Dict


class Config(object):
    def __init__(self):
        self.params = {
            # For TMC Shapley
            'tmc_iter': 500,
            'tmc_thresh': 0.001,
            # For Beta Shapley
            'beta_iter': 50,
            'alpha': 1.0,
            'beta': 16.0,
            'rho': 1.0005,
            'beta_chain': 10,
        }

    def get_params(self):
        return self.params

    def get_value(self, key: str):
        return self.params.get(key)

    def update_params(self, params: Dict):
        for (k, v) in params.items():
            if params.get(k) is None:
                raise KeyError(f"[!] Undefined key {k} with value {v}")
            else:
                self.params[k] = v

    def dump_params(self):
        print("[+] Current parameter setting:")
        print(f"{'key':<15} {'Value':<15}")
        for k, v in self.params.items():
            print(f"{k:<15} {v:<15}")