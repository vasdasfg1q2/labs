import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# цільова функція з методички
def target(x1, x2):
    return (x1 ** 2 - 8) * np.cos(x2)

# поверхня y=(x1^2-8)*cos(x2) на сітці 15x15 — рис.1 методички
n = 15
grid_x1 = np.linspace(0, 4, n)
grid_x2 = np.linspace(0, 4, n)
GX1, GX2 = np.meshgrid(grid_x1, grid_x2)
Y_TARGET = target(GX1, GX2)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(GX1, GX2, Y_TARGET, cmap="viridis", edgecolor="none")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("Target: y = (x1^2 - 8) * cos(x2)")
plt.tight_layout()
plt.savefig("outputs/task_1_target_surface.png", dpi=120)
plt.close()

# побудова FIS Мамдані
x1 = ctrl.Antecedent(np.arange(0, 4.001, 0.01), "x1")
x2 = ctrl.Antecedent(np.arange(0, 4.001, 0.01), "x2")
y = ctrl.Consequent(np.arange(-10, 10.01, 0.05), "y")

# x1: 3 трикутних терми (L, A, H) — методичка, крок 9-10, рис.2
# Params з рис.2: L = [-1.6, 0, 1.6]. Решта — рівномірно на [0,4]
# з кроком 2 (центри 0/2/4) і півшириною 1.6 (default «Add MFs» MATLAB).
x1["L"] = fuzz.trimf(x1.universe, [-1.6, 0.0, 1.6])
x1["A"] = fuzz.trimf(x1.universe, [ 0.4, 2.0, 3.6])
x1["H"] = fuzz.trimf(x1.universe, [ 2.4, 4.0, 5.6])

# x2: 5 гаусових термів (L, LA, A, HA, H) — методичка, крок 11-12, рис.3
# Params з рис.3 для L: gaussmf, σ=0.425, mean=0. Інші терми — рівномірно
# на [0,4] з центрами 0/1/2/3/4.
sigma = 0.425
x2["L"]  = fuzz.gaussmf(x2.universe, 0.0, sigma)
x2["LA"] = fuzz.gaussmf(x2.universe, 1.0, sigma)
x2["A"]  = fuzz.gaussmf(x2.universe, 2.0, sigma)
x2["HA"] = fuzz.gaussmf(x2.universe, 3.0, sigma)
x2["H"]  = fuzz.gaussmf(x2.universe, 4.0, sigma)

# y: 5 трикутних термів — методичка, крок 13-14, рис.4
# Params з рис.4 для L: [-15, -10, -5]. Решта — рівномірно на [-10,10]
# з кроком 5 (центри -10/-5/0/5/10) і півшириною 5.
y["L"]  = fuzz.trimf(y.universe, [-15, -10, -5])
y["LA"] = fuzz.trimf(y.universe, [-10,  -5,  0])
y["A"]  = fuzz.trimf(y.universe, [ -5,   0,  5])
y["HA"] = fuzz.trimf(y.universe, [  0,   5, 10])
y["H"]  = fuzz.trimf(y.universe, [  5,  10, 15])

# 9 правил дослівно з методички (с.4). Правила 7 і 9 мають однакову
# умову з різним виходом — залишено, max-агрегація відпрацює.
rules = [
    ctrl.Rule(x1["L"] & x2["L"],  y["L"]),
    ctrl.Rule(x1["L"] & x2["H"],  y["A"]),
    ctrl.Rule(x1["L"] & x2["HA"], y["H"]),
    ctrl.Rule(x1["H"] & x2["L"],  y["HA"]),
    ctrl.Rule(x1["H"] & x2["H"],  y["L"]),
    ctrl.Rule(x1["A"] & x2["A"],  y["A"]),
    ctrl.Rule(x1["A"] & x2["HA"], y["HA"]),
    ctrl.Rule(x1["L"] & x2["LA"], y["LA"]),
    ctrl.Rule(x1["A"] & x2["HA"], y["A"]),
]
system = ctrl.ControlSystem(rules)

# функції належності входів/виходу — рис.2,3,4 методички
fig, axes = plt.subplots(3, 1, figsize=(7, 8))
for term in ("L", "A", "H"):
    axes[0].plot(x1.universe, x1[term].mf, label=term)
axes[0].set_title("x1 — трикутні ФН")
axes[0].legend()
axes[0].grid(True)

for term in ("L", "LA", "A", "HA", "H"):
    axes[1].plot(x2.universe, x2[term].mf, label=term)
axes[1].set_title("x2 — гаусові ФН")
axes[1].legend()
axes[1].grid(True)

for term in ("L", "LA", "A", "HA", "H"):
    axes[2].plot(y.universe, y[term].mf, label=term)
axes[2].set_title("y — трикутні ФН")
axes[2].legend()
axes[2].grid(True)
plt.tight_layout()
plt.savefig("outputs/task_1_membership.png", dpi=120)
plt.close()

# оцінка поверхні FIS на тій же сітці 15x15
Y_FIS = np.zeros_like(Y_TARGET)
for j in range(n):
    for i in range(n):
        sim = ctrl.ControlSystemSimulation(system)
        sim.input["x1"] = float(grid_x1[i])
        sim.input["x2"] = float(grid_x2[j])
        try:
            sim.compute()
            Y_FIS[j, i] = sim.output["y"]
        except Exception:
            Y_FIS[j, i] = np.nan

# поверхня вход-вихід FIS — аналог рис.7 методички
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(GX1, GX2, Y_FIS, cmap="viridis", edgecolor="none")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("FIS surface (Mamdani, 9 rules)")
plt.tight_layout()
plt.savefig("outputs/task_1_fis_surface.png", dpi=120)
plt.close()

# тестова точка [2, 2] — відповідає рис.6 методички (Input=[2;2])
sim = ctrl.ControlSystemSimulation(system)
sim.input["x1"] = 2.0
sim.input["x2"] = 2.0
sim.compute()
y_point_fis = sim.output["y"]
y_point_target = target(2.0, 2.0)

mask = ~np.isnan(Y_FIS)
mae = float(np.mean(np.abs(Y_FIS[mask] - Y_TARGET[mask])))

print("FIS Mamdani: 2 входи (x1 — 3 трикутних, x2 — 5 гаусових), 1 вихід (y — 5 трикутних), 9 правил")
print()
print(f"Сітка оцінювання: {n}x{n} точок на [0,4]x[0,4]")
print(f"MAE(FIS vs target) = {mae:.3f}")
print()
print(f"Тестова точка x1=2, x2=2:")
print(f"  target y = {y_point_target:.3f}")
print(f"  FIS y    = {y_point_fis:.3f}")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
