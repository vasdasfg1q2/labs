import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

iris = load_iris()
X = iris["data"]
y = iris["target"]

# k=5 — як у код-підказці методички; нижче порівняємо з реальним k=3
kmeans5 = KMeans(n_clusters=5, random_state=0, n_init=10)
kmeans5.fit(X)
y_kmeans5 = kmeans5.predict(X)

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans5, s=50, cmap="viridis")
centers = kmeans5.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("KMeans (k=5) on Iris — проекція на 2 ознаки")
plt.tight_layout()
plt.savefig("outputs/task_2_kmeans5.png", dpi=120)
plt.close()


def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == j].mean(0) for j in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


centers3, labels3 = find_clusters(X, 3, rseed=2)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels3, s=50, cmap="viridis")
plt.scatter(centers3[:, 0], centers3[:, 1], c="black", s=200, alpha=0.5)
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("Власний k-means (k=3, rseed=2)")
plt.tight_layout()
plt.savefig("outputs/task_2_custom_rseed2.png", dpi=120)
plt.close()

# rseed=0 — показує залежність від ініціалізації
centers3b, labels3b = find_clusters(X, 3, rseed=0)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels3b, s=50, cmap="viridis")
plt.scatter(centers3b[:, 0], centers3b[:, 1], c="black", s=200, alpha=0.5)
plt.title("Власний k-means (k=3, rseed=0)")
plt.tight_layout()
plt.savefig("outputs/task_2_custom_rseed0.png", dpi=120)
plt.close()

kmeans3 = KMeans(n_clusters=3, random_state=0, n_init=10)
labels3_sk = kmeans3.fit_predict(X)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels3_sk, s=50, cmap="viridis")
plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1],
            c="black", s=200, alpha=0.5)
plt.title("sklearn KMeans (k=3, random_state=0) на Iris")
plt.tight_layout()
plt.savefig("outputs/task_2_sklearn_k3.png", dpi=120)
plt.close()

ari = adjusted_rand_score(y, labels3_sk)
print(f"Iris: 150 точок, 4 ознаки, 3 справжніх класи")
print(f"KMeans (k=5) inertia = {kmeans5.inertia_:.2f}")
print(f"KMeans (k=3) inertia = {kmeans3.inertia_:.2f}")
print(f"Adjusted Rand Index (k=3 vs true labels) = {ari:.4f}")
print(f"(ARI = 1 ідеал; 0 — випадкові; типове значення для Iris k=3 ≈ 0.73)")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
