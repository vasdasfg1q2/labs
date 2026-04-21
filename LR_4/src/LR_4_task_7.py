import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# варіант студента №22: ((22-1) mod 15) + 1 = 7 (методичка 1 СШІ_ЛР_4, с.3)
# X=[-12, 29, 0, 4, 6, 8], Y=[-3, 0, 1, 2, 9, 5]
X = np.array([-12.0, 29.0, 0.0, 4.0, 6.0, 8.0])
Y = np.array([-3.0, 0.0, 1.0, 2.0, 9.0, 5.0])

# МНК вручну (формули з методички 1, с.2):
#   β0 = (ΣY·ΣX² - ΣX·ΣXY) / (n·ΣX² - (ΣX)²)
#   β1 = (n·ΣXY - ΣX·ΣY)   / (n·ΣX² - (ΣX)²)
n = len(X)
sum_x = X.sum()
sum_y = Y.sum()
sum_xy = (X * Y).sum()
sum_x2 = (X * X).sum()
denom = n * sum_x2 - sum_x ** 2
beta1 = (n * sum_xy - sum_x * sum_y) / denom
beta0 = (sum_y * sum_x2 - sum_x * sum_xy) / denom

# залишкова сума квадратів
y_pred = beta0 + beta1 * X
S = ((Y - y_pred) ** 2).sum()

# перевірка через numpy.polyfit (повинні збігтися)
p = np.polyfit(X, Y, 1)
beta1_np, beta0_np = p

print(f"Варіант 7: X = {X.tolist()}, Y = {Y.tolist()}")
print()
print("Метод найменших квадратів (вручну):")
print(f"  β0 = {beta0:.4f}")
print(f"  β1 = {beta1:.4f}")
print(f"  Апроксимуюче рівняння:  y = {beta0:.3f} + {beta1:.3f}·x")
print(f"  Сума квадратів похибок S = {S:.4f}")
print()
print("Перевірка через numpy.polyfit:")
print(f"  β0 = {beta0_np:.4f}  (збіг: {'OK' if np.isclose(beta0, beta0_np) else 'FAIL'})")
print(f"  β1 = {beta1_np:.4f}  (збіг: {'OK' if np.isclose(beta1, beta1_np) else 'FAIL'})")

# графік
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X, Y, s=60, color="black", label="експериментальні точки")
xs = np.linspace(X.min() - 2, X.max() + 2, 200)
ax.plot(xs, beta0 + beta1 * xs, "r-", linewidth=2,
        label=f"y = {beta0:.2f} + {beta1:.2f}·x")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("МНК: варіант 7 (лінійна апроксимація)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/task_7_ols_variant7.png", dpi=120)
plt.close()

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
