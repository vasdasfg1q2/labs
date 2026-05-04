import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

X = np.loadtxt("data_clustering.txt", delimiter=",")
num_clusters = 5

# вхідні дані
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none",
            edgecolors="black", s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title("Input data")
plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
plt.xticks(()); plt.yticks(())
plt.tight_layout()
plt.savefig("outputs/task_1_input.png", dpi=120)
plt.close()

kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10, random_state=0)
kmeans.fit(X)

# межі кластерів через сітку
step_size = 0.01
x_vals, y_vals = np.meshgrid(
    np.arange(x_min, x_max, step_size),
    np.arange(y_min, y_max, step_size),
)
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.figure()
plt.imshow(output, interpolation="nearest",
           extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired, aspect="auto", origin="lower")
plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none",
            edgecolors="black", s=80)
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker="o", s=210, linewidths=4, color="black",
            zorder=12, facecolors="black")
plt.title("Boundaries of clusters")
plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
plt.xticks(()); plt.yticks(())
plt.tight_layout()
plt.savefig("outputs/task_1_clusters.png", dpi=120)
plt.close()

print(f"KMeans: n_clusters={num_clusters}, n_points={len(X)}")
print("Cluster centers:")
for i, c in enumerate(cluster_centers):
    print(f"  Cluster {i}: ({c[0]:.2f}, {c[1]:.2f})")
print(f"Inertia: {kmeans.inertia_:.2f}")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")
