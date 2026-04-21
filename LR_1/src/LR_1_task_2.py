import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

print("Ваги 1-го шару:")
print("  OR:  W = (1, 1), W0 = -1/2")
print("  AND: W = (1, 1), W0 = -3/2")
print("Ваги 2-го шару:")
print("  XOR: W = (+1, -1), W0 = -1/2")
print()
print("Рівняння розділяючої прямої 2-го шару: y1 - y2 = 0.5")
print()
print("(x1,x2) -> (y1,y2) -> XOR")
for x1, x2, y1, y2, y in [
    (0, 0, 0, 0, 0),
    (0, 1, 1, 0, 1),
    (1, 0, 1, 0, 1),
    (1, 1, 1, 1, 0),
]:
    print(f"({x1},{x2})    -> ({y1},{y2})   -> {y}")


# схема мережі 2 -> 2 -> 1
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

positions = {
    "x1": (1.5, 3.5),
    "x2": (1.5, 1.5),
    "y1": (5.0, 3.5),
    "y2": (5.0, 1.5),
    "y":  (8.5, 2.5),
}
labels = {"x1": "x1", "x2": "x2", "y1": "OR", "y2": "AND", "y": "XOR"}

for name, (x, y) in positions.items():
    ax.add_patch(Circle((x, y), 0.4, fill=False, linewidth=1.5))
    ax.text(x, y, labels[name], ha="center", va="center", fontsize=11)

# з'єднання
edges = [
    ("x1", "y1", "1"),
    ("x1", "y2", "1"),
    ("x2", "y1", "1"),
    ("x2", "y2", "1"),
    ("y1", "y",  "+1"),
    ("y2", "y",  "-1"),
]
for src, dst, w in edges:
    x1, y1 = positions[src]
    x2, y2 = positions[dst]
    ax.add_patch(FancyArrowPatch((x1 + 0.4, y1), (x2 - 0.4, y2),
                                  arrowstyle="->", mutation_scale=12,
                                  linewidth=1, color="black"))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, w,
            ha="center", fontsize=9)

ax.text(5.0, 4.4, "W0=-1/2", ha="center", fontsize=9)
ax.text(5.0, 0.6, "W0=-3/2", ha="center", fontsize=9)
ax.text(8.5, 1.4, "W0=-1/2", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/task_2_perceptron_scheme.png", dpi=120)
plt.close()


# розділяюча пряма в просторі (y1, y2)
fig, ax = plt.subplots(figsize=(5, 5))
points = [(0, 0, 0), (1, 0, 1), (1, 1, 0)]
for y1, y2, cls in points:
    marker = "o" if cls == 0 else "x"
    ax.plot(y1, y2, marker, markersize=12, color="black")

ys = np.linspace(-0.3, 1.3, 50)
ax.plot(ys, ys - 0.5, "-", color="black", linewidth=1,
        label="y1 - y2 = 0.5")

ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("y1 = OR")
ax.set_ylabel("y2 = AND")
ax.set_title("Розділяюча пряма 2-го шару")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("outputs/task_2_separating_line_y.png", dpi=120)
plt.close()


# розділяючі прямі першого шару у вхідному просторі (x1, x2)
# — повторює Рис.3 методички: g1 (OR) і g2 (AND)
fig, ax = plt.subplots(figsize=(5, 5))
for x1, x2, cls in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
    marker = "o" if cls == 0 else "x"
    ax.plot(x1, x2, marker, markersize=12, color="black")

xs = np.linspace(-0.3, 1.3, 50)
ax.plot(xs, 0.5 - xs, "--", color="black", linewidth=1,
        label="g1: x1 + x2 = 0.5")
ax.plot(xs, 1.5 - xs, "-",  color="black", linewidth=1,
        label="g2: x1 + x2 = 1.5")

ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Розділяючі прямі 1-го шару (OR, AND)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("outputs/task_2_separating_lines_x.png", dpi=120)
plt.close()
