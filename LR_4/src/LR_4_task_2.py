import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
from sklearn import linear_model

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# варіант студента №22: ((22-1) mod 5) + 1 = 2 → data_regr_2.txt
input_file = "data_regr_2.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
y_test_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color="green")
plt.plot(X_test, y_test_pred, color="black", linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear regression on data_regr_2.txt (variant 2)")
plt.tight_layout()
plt.savefig("outputs/task_2_variant2.png", dpi=120)
plt.close()

print("Linear regressor performance on data_regr_2.txt:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
print()
print(f"Коефіцієнти: β0 = {regressor.intercept_:.4f}, β1 = {regressor.coef_[0]:.4f}")
print(f"Рівняння прямої: y = {regressor.intercept_:.3f} + {regressor.coef_[0]:.3f}·x")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
