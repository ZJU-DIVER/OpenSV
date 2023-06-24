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
    '''
        Args:
            K (int) : KNN model parameter
            eps (float) : LSH parameter
    '''
    K: int = 16
    eps: float = 0.1

    # CS-Shapley

    #Volume-based Shapley
    """     
        Args:
            omega (float):A smaller ω means that the
            d-cubes are more refined and RV can better represent the original data instead of crudely grouping
            many data points together and representing them via a statistic. On the other hand, a larger ω means
            a less refined diversity representation but greater replication robustness
    """
    omega=0.1

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
