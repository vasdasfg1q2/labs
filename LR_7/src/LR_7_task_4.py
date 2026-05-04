import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, covariance

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# Методичка каже завантажити котирування через matplotlib.finance (видалено з
# matplotlib ≥2.2) або yahoo (API закрито Yahoo у 2017). Тому генеруємо
# реалістичні синтетичні дані: 10 компаній з 4 секторами, щоденні варіації
# (close - open) за ~250 торгових днів. Відтворюваність через seed.
np.random.seed(42)

# 10 компаній, розбитих на 4 «сектори» (tech, auto, energy, food). Компанії
# всередині сектора мають скорельовані цінові варіації.
companies = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "TSLA": "Tesla",
    "F":    "Ford",
    "XOM":  "ExxonMobil",
    "CVX":  "Chevron",
    "KO":   "Coca-Cola",
    "PEP":  "PepsiCo",
    "MCD":  "McDonalds",
}
sector_of = {
    "AAPL": 0, "MSFT": 0, "GOOGL": 0,
    "TSLA": 1, "F": 1,
    "XOM": 2, "CVX": 2,
    "KO": 3, "PEP": 3, "MCD": 3,
}

n_days = 250
symbols = list(companies.keys())
names = np.array([companies[s] for s in symbols])

# Генеруємо ціни: спільний «шум» на сектор + індивідуальний шум на компанію.
sector_noise = np.random.randn(4, n_days) * 1.5
quotes_diff = np.zeros((len(symbols), n_days))
for i, sym in enumerate(symbols):
    sec = sector_of[sym]
    quotes_diff[i] = sector_noise[sec] + np.random.randn(n_days) * 0.5

# Нормалізація (кожна колонка — відхилення від σ)
X = quotes_diff.copy().T  # (n_days, n_companies)
X /= X.std(axis=0)

# GraphicalLassoCV — sparse precision matrix; відновлює структуру зв'язків.
edge_model = covariance.GraphicalLassoCV()
with np.errstate(invalid="ignore"):
    edge_model.fit(X)

# Affinity Propagation на covariance-матриці — шукає exemplars.
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

print(f"Affinity propagation: {num_labels + 1} clusters found")
print()
for i in range(num_labels + 1):
    members = ", ".join(names[labels == i])
    print(f"Cluster {i+1} ==> {members}")

# Візуалізація: проєктуємо компанії у 2D через перші дві головні компоненти
# covariance-матриці (PCA-підхід), фарбуємо по кластерах AP.
cov = edge_model.covariance_
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
proj = cov @ eigvecs[:, order[:2]]

fig, ax = plt.subplots(figsize=(7, 5))
for i in range(num_labels + 1):
    mask = labels == i
    ax.scatter(proj[mask, 0], proj[mask, 1], s=120, label=f"Cluster {i+1}")
    for x_, y_, name in zip(proj[mask, 0], proj[mask, 1], names[mask]):
        ax.annotate(name, (x_, y_), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("Affinity Propagation: групування компаній")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/task_4_clusters.png", dpi=120)
plt.close()

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
