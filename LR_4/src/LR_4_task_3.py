import io
import os
import sys

import numpy as np
import sklearn.metrics as sm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

input_file = "data_multivar_regr.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)

print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# поліноміальна регресія ступеня 10
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
y_test_poly_pred = poly_linear_model.predict(X_test_transformed)

print()
print("Polynomial Regressor (degree=10) performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_poly_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_poly_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_poly_pred), 2))

# прогноз для вибіркової точки
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

print()
print("Прогноз для точки [[7.75, 6.35, 5.56]] (очікуване значення ~41.35):")
print("  Linear regression:     ", linear_regressor.predict(datapoint))
print("  Polynomial regression: ", poly_linear_model.predict(poly_datapoint))
