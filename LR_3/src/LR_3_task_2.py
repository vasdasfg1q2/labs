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

# входи: температура води (умовно 0..100 °C) і напір (0..100 %)
temp = ctrl.Antecedent(np.arange(0, 100.1, 0.5), "temp")
press = ctrl.Antecedent(np.arange(0, 100.1, 0.5), "press")

# виходи: кути повороту кранів на [-90, 90] градусів
hot = ctrl.Consequent(np.arange(-90, 90.1, 0.5), "hot")
cold = ctrl.Consequent(np.arange(-90, 90.1, 0.5), "cold")

# температура: 5 термів (cold, cool, warm, lukewarm, hot)
temp["cold"]     = fuzz.trimf(temp.universe, [0,    0, 25])
temp["cool"]     = fuzz.trimf(temp.universe, [10,  25, 40])
temp["warm"]     = fuzz.trimf(temp.universe, [30,  45, 60])
temp["lukewarm"] = fuzz.trimf(temp.universe, [50,  65, 80])
temp["hot"]      = fuzz.trimf(temp.universe, [70, 100, 100])

# напір: 3 терми (weak, medium, strong)
press["weak"]    = fuzz.trimf(press.universe, [0,   0,  40])
press["medium"]  = fuzz.trimf(press.universe, [20, 50,  80])
press["strong"]  = fuzz.trimf(press.universe, [60, 100, 100])

# виходи: 7 термів повороту для кожного крана
#   BL=великий вліво,  ML=середній вліво,  SL=невеликий вліво,
#   Z=нуль,
#   SR=невеликий вправо, MR=середній вправо, BR=великий вправо
for out in (hot, cold):
    out["BL"] = fuzz.trimf(out.universe, [-90, -90, -45])
    out["ML"] = fuzz.trimf(out.universe, [-60, -30,   0])
    out["SL"] = fuzz.trimf(out.universe, [-30, -15,   0])
    out["Z"]  = fuzz.trimf(out.universe, [-10,   0,  10])
    out["SR"] = fuzz.trimf(out.universe, [  0,  15,  30])
    out["MR"] = fuzz.trimf(out.universe, [  0,  30,  60])
    out["BR"] = fuzz.trimf(out.universe, [ 45,  90,  90])

# 11 правил дослівно з методички (с.6).
# Якщо для крана методичка не указує дію — ставимо Z (кран не рухається).
rules = [
    ctrl.Rule(temp["hot"]      & press["strong"], (hot["ML"], cold["MR"])),
    ctrl.Rule(temp["hot"]      & press["medium"], (hot["Z"],  cold["MR"])),
    ctrl.Rule(temp["lukewarm"] & press["strong"], (hot["SL"], cold["Z"])),
    ctrl.Rule(temp["lukewarm"] & press["weak"],   (hot["SR"], cold["SR"])),
    ctrl.Rule(temp["warm"]     & press["medium"], (hot["Z"],  cold["Z"])),
    ctrl.Rule(temp["cool"]     & press["strong"], (hot["MR"], cold["ML"])),
    ctrl.Rule(temp["cool"]     & press["medium"], (hot["MR"], cold["SL"])),
    ctrl.Rule(temp["cold"]     & press["weak"],   (hot["BR"], cold["Z"])),
    ctrl.Rule(temp["cold"]     & press["strong"], (hot["ML"], cold["MR"])),
    ctrl.Rule(temp["warm"]     & press["strong"], (hot["SL"], cold["SL"])),
    ctrl.Rule(temp["warm"]     & press["weak"],   (hot["SR"], cold["SR"])),
]
system = ctrl.ControlSystem(rules)

# графіки ФН
fig, axes = plt.subplots(4, 1, figsize=(7, 10))
for t in ("cold", "cool", "warm", "lukewarm", "hot"):
    axes[0].plot(temp.universe, temp[t].mf, label=t)
axes[0].set_title("temp — температура води")
axes[0].legend(); axes[0].grid(True)

for t in ("weak", "medium", "strong"):
    axes[1].plot(press.universe, press[t].mf, label=t)
axes[1].set_title("press — напір")
axes[1].legend(); axes[1].grid(True)

terms_out = ("BL", "ML", "SL", "Z", "SR", "MR", "BR")
for t in terms_out:
    axes[2].plot(hot.universe, hot[t].mf, label=t)
axes[2].set_title("hot — кут крана гарячої води")
axes[2].legend(ncol=4, fontsize=8); axes[2].grid(True)

for t in terms_out:
    axes[3].plot(cold.universe, cold[t].mf, label=t)
axes[3].set_title("cold — кут крана холодної води")
axes[3].legend(ncol=4, fontsize=8); axes[3].grid(True)
plt.tight_layout()
plt.savefig("outputs/task_2_membership.png", dpi=120)
plt.close()

# поверхні hot(temp, press) і cold(temp, press)
n = 15
T = np.linspace(5, 95, n)
P = np.linspace(5, 95, n)
TT, PP = np.meshgrid(T, P)
HOT = np.full_like(TT, np.nan, dtype=float)
COLD = np.full_like(TT, np.nan, dtype=float)
for j in range(n):
    for i in range(n):
        sim = ctrl.ControlSystemSimulation(system)
        sim.input["temp"] = float(T[i])
        sim.input["press"] = float(P[j])
        try:
            sim.compute()
            HOT[j, i] = sim.output.get("hot", np.nan)
            COLD[j, i] = sim.output.get("cold", np.nan)
        except Exception:
            pass

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(TT, PP, HOT, cmap="viridis", edgecolor="none")
ax1.set_xlabel("temp"); ax1.set_ylabel("press"); ax1.set_zlabel("hot, °")
ax1.set_title("hot(temp, press)")
ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(TT, PP, COLD, cmap="viridis", edgecolor="none")
ax2.set_xlabel("temp"); ax2.set_ylabel("press"); ax2.set_zlabel("cold, °")
ax2.set_title("cold(temp, press)")
plt.tight_layout()
plt.savefig("outputs/task_2_surface.png", dpi=120)
plt.close()

# тестові сценарії
cases = [
    ("гаряча + сильний",   90, 90),
    ("гаряча + середній",  85, 50),
    ("прохолодна + сил.",  25, 90),
    ("холодна + слабий",    5, 10),
    ("тепла + середній",   45, 50),
    ("тепла + слабий",     45, 10),
]
print("Нечітка модель змішувача: 11 правил, 2 входи, 2 виходи")
print()
print(f"{'ситуація':<22} {'temp':>5} {'press':>6} {'hot°':>8} {'cold°':>8}")
for name, t, p in cases:
    sim = ctrl.ControlSystemSimulation(system)
    sim.input["temp"] = t
    sim.input["press"] = p
    try:
        sim.compute()
        h = sim.output.get("hot", float("nan"))
        c = sim.output.get("cold", float("nan"))
    except Exception:
        h = c = float("nan")
    print(f"{name:<22} {t:>5} {p:>6} {h:>8.2f} {c:>8.2f}")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
