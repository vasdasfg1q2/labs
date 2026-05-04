import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

X = np.loadtxt("data_clustering.txt", delimiter=",")

# ширина вікна (bandwidth) — критичний параметр MeanShift. estimate_bandwidth
# з quantile=0.1 обере його автоматично за перцентилями попарних відстаней.
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

cluster_centers = meanshift_model.cluster_centers_
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))

print("Mean Shift clustering on data_clustering.txt")
print(f"Bandwidth (estimated): {bandwidth_X:.4f}")
print(f"Number of clusters: {num_clusters}")
print("Centers of clusters:")
for i, c in enumerate(cluster_centers):
    print(f"  Cluster {i+1}: ({c[0]:.4f}, {c[1]:.4f})")

# візуалізація точок і центрів
plt.figure()
markers = cycle("o*xvs")
for i, marker in zip(range(num_clusters), markers):
    plt.scatter(X[labels == i, 0], X[labels == i, 1],
                marker=marker, color="black")
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker="o",
             markerfacecolor="black", markeredgecolor="black", markersize=15)
plt.title(f"Mean Shift clusters (n={num_clusters})")
plt.tight_layout()
plt.savefig("outputs/task_3_meanshift.png", dpi=120)
plt.close()

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
