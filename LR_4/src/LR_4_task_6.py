import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# ті самі дані, що у task_5 (варіант 2)
np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)


def plot_learning_curves(model, X, y, title, filename):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    train_errors, val_errors = [], []
    for n in range(1, len(X_train)):
        model.fit(X_train[:n], y_train[:n])
        y_train_predict = model.predict(X_train[:n])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:n]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.figure(figsize=(7, 5))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.xlabel("training set size")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


# 1) лінійна модель
plot_learning_curves(
    LinearRegression(), X, y,
    "Learning curves: linear",
    "outputs/task_6_linear.png",
)

# 2) поліном 2 ступеня
poly2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly2, X, y,
                      "Learning curves: polynomial degree=2",
                      "outputs/task_6_poly2.png")

# 3) поліном 10 ступеня
poly10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly10, X, y,
                      "Learning curves: polynomial degree=10",
                      "outputs/task_6_poly10.png")

print("Криві навчання побудовано для трьох моделей:")
print("  - лінійна регресія            → task_6_linear.png")
print("  - поліноміальна 2 ступеня     → task_6_poly2.png  (еталонна для цих даних)")
print("  - поліноміальна 10 ступеня    → task_6_poly10.png (перенавчена)")
print()
print("Аналіз (див. DEBUG §6 task_6 для розгорнутих коментарів):")
print("  - linear: обидві криві високо і близько — недонавчена модель (bias-помилка).")
print("  - poly2:  криві низько і сходяться — добре узагальнена модель.")
print("  - poly10: великий проміжок train ↔ val — перенавчена модель (variance).")
