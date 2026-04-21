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

# входи: температура повітря в приміщенні (°C) і швидкість її зміни (°C/хв)
temperature = ctrl.Antecedent(np.arange(0, 40.01, 0.1), "temperature")
rate = ctrl.Antecedent(np.arange(-5, 5.01, 0.05), "rate")

# вихід: кут повороту регулятора кондиціонеру, вліво = холод, вправо = тепло
angle = ctrl.Consequent(np.arange(-90, 90.1, 0.5), "angle")

# температура: 5 термів (методичка, правила 1-15)
temperature["very_cold"] = fuzz.trimf(temperature.universe, [0,   5, 12])
temperature["cold"]      = fuzz.trimf(temperature.universe, [8,  13, 18])
temperature["normal"]    = fuzz.trimf(temperature.universe, [16, 21, 26])
temperature["warm"]      = fuzz.trimf(temperature.universe, [24, 28, 32])
temperature["very_warm"] = fuzz.trimf(temperature.universe, [30, 36, 40])

# швидкість зміни: 3 терми
rate["falling"] = fuzz.trimf(rate.universe, [-5, -5,   0])
rate["zero"]    = fuzz.trimf(rate.universe, [-1,  0,   1])
rate["rising"]  = fuzz.trimf(rate.universe, [ 0,  5,   5])

# кут регулятора: 7 термів на [-90, 90]
angle["BL"] = fuzz.trimf(angle.universe, [-90, -90, -45])
angle["ML"] = fuzz.trimf(angle.universe, [-60, -30,   0])
angle["SL"] = fuzz.trimf(angle.universe, [-30, -15,   0])
angle["Z"]  = fuzz.trimf(angle.universe, [-10,   0,  10])
angle["SR"] = fuzz.trimf(angle.universe, [  0,  15,  30])
angle["MR"] = fuzz.trimf(angle.universe, [  0,  30,  60])
angle["BR"] = fuzz.trimf(angle.universe, [ 45,  90,  90])

# 15 правил дослівно з методички (с.6-7).
# Правило 4: «повернути слід вимкнути» — інтерпретовано як Z (вимкнено).
# Правило 7: «тепло» + «вліво» — дослівно BL (у DEBUG §5 — пояснення парадоксу).
rules = [
    ctrl.Rule(temperature["very_warm"] & rate["rising"],  angle["BL"]),
    ctrl.Rule(temperature["very_warm"] & rate["falling"], angle["SL"]),
    ctrl.Rule(temperature["warm"]      & rate["rising"],  angle["BL"]),
    ctrl.Rule(temperature["warm"]      & rate["falling"], angle["Z"]),
    ctrl.Rule(temperature["very_cold"] & rate["falling"], angle["BR"]),
    ctrl.Rule(temperature["very_cold"] & rate["rising"],  angle["SR"]),
    ctrl.Rule(temperature["cold"]      & rate["falling"], angle["BL"]),
    ctrl.Rule(temperature["cold"]      & rate["rising"],  angle["Z"]),
    ctrl.Rule(temperature["very_warm"] & rate["zero"],    angle["BL"]),
    ctrl.Rule(temperature["warm"]      & rate["zero"],    angle["SL"]),
    ctrl.Rule(temperature["very_cold"] & rate["zero"],    angle["BR"]),
    ctrl.Rule(temperature["cold"]      & rate["zero"],    angle["SR"]),
    ctrl.Rule(temperature["normal"]    & rate["rising"],  angle["SL"]),
    ctrl.Rule(temperature["normal"]    & rate["falling"], angle["SR"]),
    ctrl.Rule(temperature["normal"]    & rate["zero"],    angle["Z"]),
]
system = ctrl.ControlSystem(rules)

# графіки ФН
fig, axes = plt.subplots(3, 1, figsize=(7, 8))
for t in ("very_cold", "cold", "normal", "warm", "very_warm"):
    axes[0].plot(temperature.universe, temperature[t].mf, label=t)
axes[0].set_title("temperature — температура повітря (°C)")
axes[0].legend(); axes[0].grid(True)

for t in ("falling", "zero", "rising"):
    axes[1].plot(rate.universe, rate[t].mf, label=t)
axes[1].set_title("rate — швидкість зміни (°C/хв)")
axes[1].legend(); axes[1].grid(True)

for t in ("BL", "ML", "SL", "Z", "SR", "MR", "BR"):
    axes[2].plot(angle.universe, angle[t].mf, label=t)
axes[2].set_title("angle — кут регулятора (вліво = холод, вправо = тепло)")
axes[2].legend(ncol=4, fontsize=8); axes[2].grid(True)
plt.tight_layout()
plt.savefig("outputs/task_3_membership.png", dpi=120)
plt.close()

# поверхня angle(temperature, rate)
n = 21
T = np.linspace(2, 38, n)
R = np.linspace(-4, 4, n)
TT, RR = np.meshgrid(T, R)
ANG = np.full_like(TT, np.nan, dtype=float)
for j in range(n):
    for i in range(n):
        sim = ctrl.ControlSystemSimulation(system)
        sim.input["temperature"] = float(T[i])
        sim.input["rate"] = float(R[j])
        try:
            sim.compute()
            ANG[j, i] = sim.output.get("angle", np.nan)
        except Exception:
            pass

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(TT, RR, ANG, cmap="viridis", edgecolor="none")
ax.set_xlabel("temperature, °C")
ax.set_ylabel("rate, °C/хв")
ax.set_zlabel("angle, °")
ax.set_title("angle(temperature, rate) — режим кондиціонеру")
plt.tight_layout()
plt.savefig("outputs/task_3_surface.png", dpi=120)
plt.close()

# тестова таблиця сценаріїв
cases = [
    ("дуже тепло, росте",   33,  2.5),
    ("тепло, падає",        27, -2.5),
    ("в нормі, стабільно",  21,  0.0),
    ("холодно, падає",      12, -2.0),
    ("дуже холодно, росте",  5,  2.0),
    ("холодно, росте",      13,  2.0),
    ("тепло, стабільно",    27,  0.0),
]
print("Нечіткий керуючий контур кондиціонера: 15 правил, 2 входи, 1 вихід")
print("Конвенція: кут < 0 — режим ХОЛОД, кут > 0 — режим ТЕПЛО, кут ≈ 0 — вимкнено")
print()
print(f"{'сценарій':<22} {'t°C':>5} {'rate':>6} {'angle°':>8}")
for name, t, r in cases:
    sim = ctrl.ControlSystemSimulation(system)
    sim.input["temperature"] = t
    sim.input["rate"] = r
    try:
        sim.compute()
        a = sim.output.get("angle", float("nan"))
    except Exception:
        a = float("nan")
    print(f"{name:<22} {t:>5} {r:>6.1f} {a:>8.2f}")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
