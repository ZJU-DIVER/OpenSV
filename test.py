from sklearn.datasets import *

data = load_iris()
X = data.data
y = data.target

X_train, X_valid, y_train, y_valid = X[:120], X[120:], y[:120], y[120:]

import opensv 

shap = opensv.DataShapley()
shap.load(X_train, y_train, X_valid, y_valid, para_tbl=opensv.ParamsTable(num_perm=5))
shap.solve()
print(shap.get_values())