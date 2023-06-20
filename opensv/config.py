from dataclasses import dataclass, fields


@dataclass
class ParamsTable:
    # ====== Values ======
    # Data Shapley

    # Beta Shapley
    beta_beta: float = 1.0
    beta_alpha: float = 16.0
    beta_rho: float = 1.0005
    beta_chain: int = 10
    
    #KNN Shapley
    K: int = 16
    eps: float = 0.1

    # CS-Shapley

    #Volume-based Shapley

    # TODO: more values
    
    # ====== Solutions ======
    num_perm: int = 500
    truncated_threshold: float = 0.001
    
    # TODO more solutions
    def __repr__(self):
        title = f"| {'Key':<25}| {'Value':<15}|\n"
        content = ""
        for field in fields(self):
            content += f"| {field.name:<25}| {getattr(self, field.name):<15}|\n"
        return ("+" + "-"*43 + "+\n" + title + 
                "|" + "-"*43 + "|\n" + content + 
                "+" + "-"*43 + "+")
