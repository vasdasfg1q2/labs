import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# завдання 3 методички 1: інтерполяція 5 точок, поліном 4 ступеня.
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])
n = len(x)
m = 4  # степінь полінома (за завданням)

# матриця Вандермонда: X[i,j] = x_i^j для j = 0..m
X = np.vstack([x ** j for j in range(m + 1)]).T
# розв'язуємо XA = y
A = np.linalg.solve(X, y)

print("Коефіцієнти інтерполяційного полінома (від a0 до a4):")
for j, a in enumerate(A):
    print(f"  a{j} = {a:.6f}")
print()
eq = " + ".join(f"{a:.4f}·x^{j}" if j else f"{a:.4f}" for j, a in enumerate(A))
print(f"P(x) = {eq}")


def P(xx):
    return sum(a * xx ** j for j, a in enumerate(A))


# значення у проміжних точках 0.2 і 0.5 (вимога завдання)
for x0 in (0.2, 0.5):
    print(f"P({x0}) = {P(x0):.6f}")

# графік
xs = np.linspace(x.min() - 0.05, x.max() + 0.05, 300)
ys = P(xs)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x, y, s=80, color="black", label="вузли інтерполяції", zorder=3)
ax.plot(xs, ys, "r-", linewidth=2, label="P(x), степінь 4")
ax.scatter([0.2, 0.5], [P(0.2), P(0.5)],
           s=80, facecolors="none", edgecolors="blue",
           linewidth=2, label="P(0.2), P(0.5)", zorder=4)
ax.set_xlabel("x")
ax.set_ylabel("P(x)")
ax.set_title("Інтерполяційний поліном 4 ступеня")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/task_8_interp.png", dpi=120)
plt.close()

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
