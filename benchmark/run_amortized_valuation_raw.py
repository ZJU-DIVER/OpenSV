# This is a test
# TODO: Make it a benchmark

import sys

sys.path.append("..")

from sklearn.datasets import *

data = load_iris()
X = data.data
y = data.target

X_train, X_valid, y_train, y_valid = X[:120], X[120:], y[:120], y[120:]

from opensv import AmortizedValuationRaw

shap = AmortizedValuationRaw()
shap.load(X_train, y_train, X_valid, y_valid)
shap.solve()
print(shap.get_values())
