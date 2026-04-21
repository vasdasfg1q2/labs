import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# варіант студента №22: ((22-1) mod 10) + 1 = 2
# m=100; X = 6*rand(m,1) - 3; y = 0.6*X^2 + X + 2 + randn(m,1)
np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)

# лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin = lin_reg.predict(X)

# поліноміальна ступеня 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(f"X[0]      = {X[0]}")
print(f"X_poly[0] = {X_poly[0]}  (X, X²)")

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# прогноз для плавного діапазону
X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_poly = poly_reg.predict(X_plot_poly)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X, y, s=15, color="steelblue", label="дані")
ax.plot(X_plot, lin_reg.predict(X_plot), "b--", linewidth=2, label="лінійна регресія")
ax.plot(X_plot, y_poly, "r-", linewidth=2, label="поліноміальна (deg=2)")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Variant 2: y = 0.6·X² + X + 2 + гаусів шум")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/task_5_selfmade.png", dpi=120)
plt.close()

print()
print("Еталонна модель:   y = 0.6·X² + 1.0·X + 2.0 + noise")
print(f"Лінійна регресія:  y = {lin_reg.intercept_[0]:.2f} + {lin_reg.coef_[0][0]:.2f}·X")
print(f"Поліноміальна:     y = {poly_reg.intercept_[0]:.2f} + "
      f"{poly_reg.coef_[0][0]:.2f}·X + {poly_reg.coef_[0][1]:.2f}·X²")
print()
print("Коефіцієнти поліноміальної моделі близькі до еталонних (0.6, 1, 2)?")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
