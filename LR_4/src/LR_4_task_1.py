import io
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
from sklearn import linear_model

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

input_file = "data_singlevar_regr.txt"
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
plt.title("Linear regression on data_singlevar_regr.txt")
plt.tight_layout()
plt.savefig("outputs/task_1_singlevar.png", dpi=120)
plt.close()

print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

output_model_file = "outputs/task_1_model.pkl"
with open(output_model_file, "wb") as f:
    pickle.dump(regressor, f)

# завантаження моделі з файла (виправлено пастку методички)
with open(output_model_file, "rb") as f:
    regressor_model = pickle.load(f)
y_test_pred_new = regressor_model.predict(X_test)
print()
print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
