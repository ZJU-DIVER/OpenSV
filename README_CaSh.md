# CaSh: Shapley Value Computation with Cache Optimization

Welcome to the specific documentation for **CaSh**. CaSh is a high-performance, algorithm-agnostic caching framework designed to accelerate sampling-based Shapley value approximation by eliminating redundant coalition evaluations via **Direct Memory Mapping** and **Dense Indexing**.

This document provides focused instructions on how to adopt the CaSh optimization in your own data valuation/interpretability pipelines, and how to reproduce the end-to-end Python experiments presented in our paper.

> **Note on C++ Micro-benchmarks**: If you are looking to reproduce the bare-metal C++ micro-benchmarks (Section 5.2 of the paper) which compare the micro-architectural overhead of Direct Mapping against `std::unordered_map`, please refer to the `cpp_micro_benchmarks/` directory.

---

## 1. What Does CaSh Do?

Existing Shapley value approximation algorithms (like Monte Carlo Permutation Sampling) inevitably re-evaluate identical feature subsets (coalitions) across different sampling iterations. 

By injecting the **CaSh** middleware, the system intercepts these queries. It uses a mathematically derived limit (`L`) to cache the highly-redundant small and large coalitions in a contiguous memory block ($O(1)$ access, zero hash-collisions), bypassing the expensive utility function (e.g., ML model retraining) with strictly zero approximation error.

---

## 2. Code Adoption: How to Enable CaSh

In this repository, CaSh is tightly integrated into the core `DataShapley` pipeline. We use a **naming convention** to distinguish whether CaSh is enabled for a specific algorithm:

*   **Baseline Methods (Uncached)**: Methods without the `_mem_` tag.
    *   Example: `"monte_carlo_ulimit_mp"`
*   **CaSh-Accelerated Methods**: Methods containing the `_mem_` tag.
    *   Example: `"monte_carlo_mem_ulimit_mp"`

### Quick Start API

To enable CaSh, you simply need to specify the `_mem_` method name and pass the cache limit `L` through the `mem_param` argument in `ParamsTable`.

```python
from opensv import DataShapley
from opensv.config import ParamsTable

# 1. Define caching parameters
# mem_param =[L, hash_table_limit, enable_complementary_cache]
# Setting L=5 is the recommended optimal configuration for n=30 players.
cache_limit_L = 5 
para_table = ParamsTable(
    num_proc=1, 
    num_utility=10000,          # Total sampling budget
    mem_param=[cache_limit_L, -1, True]  # Enalbe CaSh up to size L
)

# 2. Load dataset and model
shap = DataShapley()
shap.load(x_train, y_train, x_val, y_val, clf=your_ml_model, para_tbl=para_table)

# 3. Solve using CaSh-enabled method
# Replace "monte_carlo_ulimit_mp" with "monte_carlo_mem_ulimit_mp"
shap.solve("monte_carlo_mem_ulimit_mp")

# 4. Retrieve values
shapley_values = shap.get_values()