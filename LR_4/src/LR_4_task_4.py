import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest)

print("Linear regression on load_diabetes (10 features, 442 samples)")
print(f"Coefficients (10):  {np.round(regr.coef_, 2)}")
print(f"Intercept (β0):     {regr.intercept_:.4f}")
print(f"R2 score:           {r2_score(ytest, ypred):.4f}")
print(f"Mean absolute error:{mean_absolute_error(ytest, ypred):.4f}")
print(f"Mean squared error: {mean_squared_error(ytest, ypred):.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
ax.set_xlabel("Виміряно")
ax.set_ylabel("Передбачено")
ax.set_title("Diabetes: measured vs predicted")
plt.tight_layout()
plt.savefig("outputs/task_4_diabetes.png", dpi=120)
plt.close()

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
