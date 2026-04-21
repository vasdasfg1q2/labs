import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def step(v):
    return 1 if v >= 0 else 0


def neuron(x, w, w0):
    return step(np.dot(w, x) + w0)


def or_gate(x1, x2):
    return neuron([x1, x2], [1, 1], -0.5)


def and_gate(x1, x2):
    return neuron([x1, x2], [1, 1], -1.5)


def xor_gate(x1, x2):
    y1 = or_gate(x1, x2)
    y2 = and_gate(x1, x2)
    return neuron([y1, y2], [1, -1], -0.5)


print("Ваги нейронів:")
print("  OR:  W = (1, 1), поріг = 1/2")
print("  AND: W = (1, 1), поріг = 3/2")
print("  XOR (2-й шар): W = (1, -1), поріг = 1/2")
print()
print("Таблиця істинності:")
print(" x1 x2 | OR AND | XOR")
print("-" * 22)
for x1 in (0, 1):
    for x2 in (0, 1):
        y1 = or_gate(x1, x2)
        y2 = and_gate(x1, x2)
        y = xor_gate(x1, x2)
        print(f"  {x1}  {x2} |  {y1}  {y2}  |  {y}")

expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
for (x1, x2), want in expected.items():
    assert xor_gate(x1, x2) == want
print()

os.makedirs("outputs", exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 5))
for x1 in (0, 1):
    for x2 in (0, 1):
        y = xor_gate(x1, x2)
        marker = "o" if y == 0 else "x"
        ax.plot(x1, x2, marker, markersize=12, color="black")
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("XOR: o — клас 0, x — клас 1")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("outputs/task_1_truth_table.png", dpi=120)
plt.close()
